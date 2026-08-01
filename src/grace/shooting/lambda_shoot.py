# Import packages:
import numpy as np
from .newton_shoot import newton_shoot
from .bounds import expand_bounds, project_box, expand_weights, weighted_cost, weighted_cost_grad, safe_solve

# Normalize a radius argument into one radius per obstacle:
def set_radii(R, n):

    # Ensure radius was given:
    R_val = np.atleast_1d(np.asarray(R, float)).ravel()

    # If scalar, apply the same radius to all obstacles:
    if R_val.size == 1:
        return np.full(n, float(R_val[0]))

    # If not scalar, ensure the size matches:
    if R_val.size != n:
        raise ValueError("[GRACE] ERROR: R must be a scalar or one radius per obstacle ")

    # Return radius for all obstacles:
    return R_val

# Project obstacle onto segments to catch obstacle violations between nodes:
def segment_distances(p, o):

    # Compute distances of node positions wrt the obstacle:
    d_node = np.sqrt(np.sum((p - o) ** 2, axis=1) + 1e-12)

    # Compute change in position for each step to get direction:
    pos_beg = p[:-1]
    diff_pos = p[1:] - pos_beg

    # Normalization term:
    den = np.sum(diff_pos * diff_pos, axis=1) + 1e-12

    # Fraction along each segment where the obstacle projects onto it:
    t = np.clip(np.sum((o - pos_beg) * diff_pos, axis=1) / den, 0.0, 1.0)

    # The closest point on each segment, which generally lies between two nodes:
    foot = pos_beg + t[:, None] * diff_pos

    # Distance from the obstacle to those closest points:
    d_seg = np.sqrt(np.sum((foot - o) ** 2, axis=1) + 1e-12)

    # Score each node by the closer of itself and the segment leading into it:
    d = d_node.copy()
    d[1:] = np.minimum(d[1:], d_seg)

    # Return the path distances, and where along each segment the closest point
    # fell, which the gradient needs to blend the two node Jacobians:
    return d, np.concatenate([[0.0], t])

# Outward normal at the closest point, with the symmetric case handled:
def outward_normal(foot, o, p, k, R):

    # Radial direction from the obstacle centre out to the closest point:
    radial = foot - o
    dist = np.linalg.norm(radial)

    # Away from the centre this is well defined and is the direction to push:
    if dist >= 1e-6 * max(R, 1.0):
        return radial / dist

    # Helps break symmetry saddle point:
    k_seg = max(k, 1)
    travel = p[k_seg] - p[k_seg - 1]
    travel_norm = np.linalg.norm(travel)

    # A stationary segment has no direction of travel to work from, so fall back
    # to a fixed lateral direction rather than dividing by nothing:
    if travel_norm < 1e-12:
        return np.array([0.0, 1.0])

    # Perpendicular to the direction of travel:
    return np.array([-travel[1], travel[0]]) / travel_norm

# Roll out and evaluate the obstacle penalty at the current control:
def augmented_lagrange(system, U, OBS, radii, mu, rho, tidx, zt):

    # Fetch trajectory:
    Z = np.asarray(system.rollout(U))

    # Pull position if there is an obstacle:
    obs_pos = Z[:, list(system.pos_idx)] if OBS else None

    # Initialize metrics and loop through obstacles:
    penalty = 0.0
    worst_viol = -np.inf
    for i, o in enumerate(OBS):

        # Compute violations:
        d, _ = segment_distances(obs_pos, o)
        h = radii[i] - d
        worst_viol = max(worst_viol, float(h.max()))

        # Shifted multiplier - the current estimate of the Lagrange multiplier on
        # this constraint, which is zero wherever the constraint is inactive:
        lam_obs = np.maximum(0.0, mu[i] + rho * h)

        # Add penalty:
        penalty += float(np.sum(lam_obs ** 2 - mu[i] ** 2)) / (2.0 * rho)

    # Return trajectory, positions, penalty, worst violation, and endpoint errors:
    return Z, obs_pos, penalty, worst_viol, np.linalg.norm(Z[-1][tidx] - zt)

# Assemble the augmented-Lagrangian gradient and the active constraint rows:
def augmented_lagrange_grad(system, p, Jp, OBS, radii, mu, rho, g, n_dof):

    # Initialize list and loop through obstacles:
    con_rows = []
    for i, o in enumerate(OBS):

        # Compute violations and the shifted multipliers they produce:
        d, t_seg = segment_distances(p, o)
        lam_obs = np.maximum(0.0, mu[i] + rho * (radii[i] - d))
        active = np.where(lam_obs > 0.0)[0]

        # If no violations, continue:
        if len(active) == 0:
            continue

        # Initialize lists:
        jac_foot = np.empty((len(active), 2, n_dof))
        normals = np.empty((len(active), 2))

        # Loop through each violation:
        for j, k in enumerate(active):

            # If violation is not on segment, just use k-th information:
            if k == 0 or t_seg[k] <= 0.0:
                jac_foot[j] = Jp[k]
                foot = p[k]

            # If violation lies on segment between two nodes:
            else:

                # Compute adjusted information using the data from the two segment nodes:
                w = t_seg[k]
                jac_foot[j] = (1.0 - w) * Jp[k - 1] + w * Jp[k]
                foot = (1.0 - w) * p[k - 1] + w * p[k]

            # Direction the constraint pushes at that point:
            normals[j] = outward_normal(foot, o, p, k, radii[i])

        # Rows of the constraint gradient, and their contribution to the cost:
        grad_h = -np.einsum('kj,kjd->kd', normals, jac_foot)
        g += lam_obs[active] @ grad_h
        con_rows.append(grad_h)

    # Return the augmented gradient and the stacked active constraint rows:
    return g, (np.vstack(con_rows) if con_rows else np.zeros((0, n_dof)))

# Minimum-effort shoot to a target, optionally avoiding circular obstacles:
def lambda_shoot(system, z_target, obstacles=None, R=None, U0=None,
                 u_lo=None, u_hi=None, R_weights=None, max_it=1200, ftol=1e-6,
                 rho0=10.0, outer=25, inner=60, tol_c=1e-3, jac_reuse=5,
                 depth=8, reg=1e-10):

    # Only focus on constrained states (i.e., ones we are focusing on for endpoint):
    zt = system.target(z_target)

    # Extract number of constriained states:
    m = system.m
    tidx = list(system.tidx)
    n_dof = system.N * system.nu

    # Expand bounds to full trajectory, build weight matrix:
    lo, hi = expand_bounds(u_lo, u_hi, system.N, system.nu)
    r_diag = expand_weights(R_weights, system.N, system.nu)

    # Hessian of the effort term and its inverse.  With no weighting this is
    # twice the identity and every expression below collapses to the plain
    # Gramian form:
    hess = 2.0 * r_diag
    hess_inv = 1.0 / hess

    # Gather obstacle geometry.  With none, the loop below is exactly the
    # unconstrained lambda shoot:
    OBS = [np.asarray(o, float) for o in obstacles] if obstacles else []
    radii = set_radii(R, len(OBS)) if OBS else np.zeros(0)

    # The obstacle rows are built from the position Jacobian, so a system built
    # without pos_idx cannot avoid anything:
    if OBS and getattr(system, "pos_jac", None) is None:
        raise ValueError("[GRACE] ERROR: obstacle avoidance needs a position Jacobian "
                         "-- rebuild the system with pos_idx set")

    # Newton shoot for feasible control sequence:
    U = newton_shoot(system, zt, U0, u_lo=u_lo, u_hi=u_hi)

    # A non-finite feasibility shoot leaves nothing to optimize from:
    if not np.all(np.isfinite(U)):
        system._infeasible = True
        print("[GRACE] WARNING: target does not appear reachable. "
              "Try a longer horizon N, more time, or a closer target.")
        return U

    # Multipliers, one per node per obstacle, and the penalty weight:
    mu = np.zeros((len(OBS), system.N + 1))
    rho = rho0
    prev_worst = np.inf

    # Best iterate seen so far, so a step that leaves the well behaved region can
    # always be rolled back:
    U_best = U.copy()
    best_step = np.inf

    # Outer augmented-Lagrangian rounds.  With no obstacles there is nothing to
    # update, so a single pass is the whole solve:
    for outer_it in range(outer if OBS else 1):

        # History for the acceleration, and the trust in the mixed step:
        Jp = None
        hist_U = []
        hist_F = []
        blend = 1.0
        prev_step = np.inf

        # Inner solve at fixed multipliers:
        for inner_it in range(inner if OBS else max_it):

            # A non-finite iterate means the last step left the region where the
            # dynamics integrate, so fall back to the best point recorded:
            if not np.all(np.isfinite(U)):
                U = U_best
                break

            # Roll out and evaluate the penalty at the current control:
            Z, p, penalty, worst_viol, _ = augmented_lagrange(
                system, U, OBS, radii, mu, rho, tidx, zt)
            if not np.all(np.isfinite(Z)):
                U = U_best
                break

            # Recompute endpoint jacobian and residual:
            e, Co = system.endpoint_jac(U)
            if OBS and (Jp is None or inner_it % jac_reuse == 0):
                Jp = np.array(system.pos_jac(U))
            r = Z[-1][tidx] - zt

            # Cost gradient, with the obstacle terms folded in:
            g = weighted_cost_grad(U, r_diag)
            con_rows = np.zeros((0, n_dof))
            if OBS:
                g, con_rows = augmented_lagrange_grad(
                    system, p, Jp, OBS, radii, mu, rho, g, n_dof)

            # Inverse of H = hess + rho C'C applied by the Woodbury identity, which
            # turns an n_dof by n_dof system into one the size of the active set:
            if con_rows.shape[0]:
                n_act = con_rows.shape[0]
                mid = np.eye(n_act) / rho + (con_rows * hess_inv) @ con_rows.T \
                    + 1e-12 * np.eye(n_act)

                def apply_hess_inv(V, C=con_rows, Mid=mid):
                    corr = C.T @ safe_solve(Mid, (C * hess_inv) @ V)
                    return (hess_inv[:, None] * (V - corr)) if V.ndim > 1 \
                        else hess_inv * (V - corr)

            # With no active rows the Hessian is just the effort term:
            else:

                def apply_hess_inv(V):
                    return (hess_inv[:, None] * V) if V.ndim > 1 else hess_inv * V

            # Weighted controllability Gramian and the lambda step.  With no
            # obstacles H is the effort Hessian and this reduces exactly to the
            # least-norm control satisfying the linearized endpoint constraint:
            Hi_Co = apply_hess_inv(Co.T)
            Gram = Co @ Hi_Co + reg * np.eye(m)
            Hi_g = apply_hess_inv(g)
            F = -(Hi_g - Hi_Co @ safe_solve(Gram, Co @ Hi_g - r))
            step_norm = np.linalg.norm(F)
            if not np.isfinite(step_norm):
                U = U_best
                break

            # Record the best iterate before the stopping test, so a stopped or
            # wandering tail never loses progress already made:
            if step_norm < best_step:
                best_step = step_norm
                U_best = U.copy()

            # Stop once the step is small relative to the control itself:
            if step_norm < ftol * max(np.linalg.norm(U), 1.0):
                break

            # With no obstacles the step is a fixed-point map:
            if not OBS:

                # A growing step means the mixing has left the region where it is
                # valid, so drop the history and damp until it recovers:
                if step_norm > prev_step:
                    hist_U.clear()
                    hist_F.clear()
                    blend = max(blend * 0.5, 1e-3)
                else:
                    blend = min(blend * 1.2, 1.0)
                prev_step = step_norm

                # Append to the history and keep only the most recent entries:
                hist_U.append(U.copy())
                hist_F.append(F.copy())
                if len(hist_U) > depth:
                    hist_U.pop(0)
                    hist_F.pop(0)

                # Damped step of the plain map on the first pass, mixed step after:
                n_hist = len(hist_U)
                if n_hist == 1:
                    U_new = U + blend * F
                else:
                    dF = np.column_stack(
                        [hist_F[i + 1] - hist_F[i] for i in range(n_hist - 1)])
                    dU = np.column_stack(
                        [hist_U[i + 1] - hist_U[i] for i in range(n_hist - 1)])
                    try:
                        gamma = np.linalg.lstsq(dF, F, rcond=1e-10)[0]
                        U_new = U + blend * F - (dU + blend * dF) @ gamma
                    except np.linalg.LinAlgError:
                        U_new = U + blend * F

                # Reject a mixed step that is non-finite or far longer than the plan step:
                if not np.all(np.isfinite(U_new)) or \
                        np.linalg.norm(U_new - U) > 50.0 * max(step_norm, 1e-12):
                    hist_U.clear()
                    hist_F.clear()
                    blend = max(blend * 0.5, 1e-3)
                    U_new = U + blend * F

                # Project the trial step into the box constraints:
                U = project_box(U_new, lo, hi)
                continue

            # Line search on the augmented Lagrangian merit:
            merit = weighted_cost(U, r_diag) + penalty + 1e4 * np.linalg.norm(r)
            moved = False
            for a in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
                U_try = project_box(U + a * F, lo, hi)
                _, _, penalty_try, _, err_try = augmented_lagrange(
                    system, U_try, OBS, radii, mu, rho, tidx, zt)
                if not np.isfinite(penalty_try + err_try):
                    continue
                if weighted_cost(U_try, r_diag) + penalty_try + 1e4 * err_try < merit:
                    U = U_try
                    moved = True
                    break

            # No step size lowered the merit, so this round is finished:
            if not moved:
                break

        # With no obstacles the inner solve is the whole answer:
        if not OBS:
            U = U_best
            break

        # Multiplier update, and the complementarity measure:
        Z, p, _, worst_viol, _ = augmented_lagrange(
            system, U, OBS, radii, mu, rho, tidx, zt)
        slack = 0.0
        for i, o in enumerate(OBS):
            d, _ = segment_distances(p, o)
            mu[i] = np.maximum(0.0, mu[i] + rho * (radii[i] - d))
            slack = max(slack, float(np.max(mu[i] * np.maximum(0.0, d - radii[i]))))

        # At least one multiplier update has to happen before the outer loop can exit:
        if outer_it > 0 and worst_viol < 1e-7 and slack < tol_c:
            break

        # Tighten the penalty only if the worst violation did not fall enough:
        if worst_viol > 0.5 * prev_worst:
            rho *= 4.0
        prev_worst = worst_viol

    # Clean up feasibility:
    U = newton_shoot(system, zt, U, u_lo=u_lo, u_hi=u_hi)

    # Flag if the endpoint could not be reached:
    end_err = np.linalg.norm(system.endpoint(U) - zt)
    system._infeasible = bool(end_err > 1e-2)

    # Flag if an obstacle is still penetrated:
    if OBS:
        p = np.asarray(system.rollout(U))[:, list(system.pos_idx)]
        margin = min((segment_distances(p, o)[0] - radii[i]).min()
                     for i, o in enumerate(OBS))
        system._infeasible = bool(system._infeasible or margin < -1e-2)

    # Warn if the request could not be met:
    if system._infeasible:
        print("[GRACE] WARNING: target does not appear reachable. "
              "Try a longer horizon N, more time, or a closer target.")

    # Return control:
    return U