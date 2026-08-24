# Import packages:
import numpy as np
import casadi as ca
from .newton_shoot import newton_shoot
from .bounds import expand_weights, weighted_cost, weighted_cost_grad, safe_solve

# Compile a list of constraint expressions into evaluators:
def compile_constraints(system, funcs):

    # Nothing to compile:
    if not funcs:
        return None

    # Symbolic state, control, and one evaluation per node:
    z = ca.MX.sym("z", system.nx)
    u = ca.MX.sym("u", system.nu)
    K = system.N + 1

    # One block per constraint:
    blocks = []

    # Loop through each function:
    for f in funcs:

        # Evaluate the expression symbolically:
        expr = f(z, u)

        # Only the states and controls it touches are differentiated:
        s_dep = [i for i, b in enumerate(ca.which_depends(expr, z, 1, False)) if b]
        u_dep = [i for i, b in enumerate(ca.which_depends(expr, u, 1, False)) if b]

        # Local gradients, before any chaining through the trajectory:
        Jz = ca.jacobian(expr, z)
        Ju = ca.jacobian(expr, u)

        # Value and gradient separately, since values are asked for far more often:
        g_val = ca.Function("g", [z, u], [expr])
        g_jac = ca.Function("gj", [z, u],
                            [Jz[:, s_dep] if s_dep else ca.MX.zeros(1, 0),
                             Ju[:, u_dep] if u_dep else ca.MX.zeros(1, 0)])

        # Map over every node and store:
        f_val = ca.Function.map(g_val, K)
        f_jac = ca.Function.map(g_jac, K)
        blocks.append((s_dep, u_dep, f_val, f_jac))

    # Union of the states any constraint needs, so one row Jacobian serves all:
    states = tuple(sorted({i for b in blocks for i in b[0]}))
    return blocks, states, K

# Evaluate every constraint at a control sequence:
def eval_constraints(system, con, U, want_jac=False):

    # Unpack and roll the trajectory out:
    blocks, states, K = con
    N, nu = system.N, system.nu
    Z = np.asarray(system.rollout(U), float)

    # The terminal node has no control of its own, so it reuses the last one:
    idx = [min(k, N - 1) for k in range(K)]
    Un = np.stack([U[i * nu:(i + 1) * nu] for i in idx], axis=1)
    JZ = system.row_jac(U, states) if (want_jac and states) else None

    # Values and gradient rows, block by block:
    h_all = []
    J_all = []

    # Loop through each block:
    for s_dep, u_dep, f_val, f_jac in blocks:

        # Values only, if that is all that was asked:
        h_all.append(np.array(f_val(Z.T, Un)).flatten())
        if not want_jac:
            continue

        # Local gradients at every node:
        gz, gu = f_jac(Z.T, Un)

        # Chain the state part with dz/dU:
        rows = np.zeros((K, N * nu))
        if s_dep:
            cols = [states.index(i) for i in s_dep]
            gz = np.array(gz).reshape(K, len(s_dep))
            rows += np.einsum("kj,kjd->kd", gz, JZ[:, cols, :])

        # A control affects only its own node, so its gradient is placed directly:
        if u_dep:
            gu = np.array(gu).reshape(K, len(u_dep))
            for k in range(K):
                base = idx[k] * nu
                for c, j in enumerate(u_dep):
                    rows[k, base + j] += gu[k, c]

        # Store this block's rows:
        J_all.append(rows)

    # Stack the blocks into one h and one Jacobian:
    h = np.concatenate(h_all) if h_all else np.zeros(0)
    return h, (np.vstack(J_all) if (want_jac and J_all) else None)

# Nudge a starting trajectory off a symmetric saddle:
def _break_symmetry(system, con, U, n_dof, rounds=6):

    # Nothing to nudge if no constraint reads the states:
    blocks, states, K = con
    if not states:
        return U

    # Nudge repeatedly, since how far to move depends on how deep it started:
    for _ in range(rounds):

        # Current values and gradients:
        h, J = eval_constraints(system, con, U, want_jac=True)

        # Nothing violated, nothing to escape:
        if h.size == 0:
            return U
        viol = h > 0.0
        if not viol.any():
            return U

        # Violated rows whose gradient has vanished, judged against those that have not:
        rn = np.linalg.norm(J, axis=1)
        ref = np.median(rn[viol])
        stuck_rows = viol & (rn < 0.05 * max(ref, 1e-12))
        if not stuck_rows.any():
            return U

        # Geometry at the middle of the stuck run:
        Z = np.asarray(system.rollout(U))
        JZ = system.row_jac(U, states)
        nodes = np.unique(np.where(stuck_rows)[0] % K)
        k = int(nodes[len(nodes) // 2])
        k_prev = max(k - 1, 0)

        # How far the endpoint can move each state at this node:
        reach = np.linalg.norm(JZ[k], axis=1)
        if reach.max() <= 0.0:
            return U

        # Escape sideways: along the path does not leave a region centred on it:
        travel = Z[k, list(states)] - Z[k_prev, list(states)]
        t_norm = np.linalg.norm(travel)
        direction = reach / reach.max()
        if t_norm > 1e-12:
            unit = travel / t_norm
            direction = direction - float(direction @ unit) * unit

        # Perfectly symmetric leaves nothing to prefer, so take any perpendicular:
        if np.linalg.norm(direction) < 1e-9:
            if t_norm > 1e-12 and len(states) >= 2:
                direction = np.zeros(len(states))
                direction[0], direction[1] = -travel[1], travel[0]
            else:
                direction = np.zeros(len(states))
                direction[int(np.argmax(reach))] = 1.0

        # Fixed sign, so repeated solves of the same problem agree:
        n = np.linalg.norm(direction)
        if n < 1e-12:
            return U
        direction = direction / n
        if direction[int(np.argmax(np.abs(direction)))] < 0.0:
            direction = -direction

        # Move every stuck node, since shifting one leaves the rest on the saddle:
        Jk = np.zeros(U.size)
        for kk in nodes:
            Jk = Jk + np.einsum("j,jd->d", direction, JZ[int(kk)])
        jn = float(Jk @ Jk)
        if jn < 1e-18:
            return U

        # h for a keep-out is an area, so its root is the distance still to travel:
        depth = float(np.sqrt(max(h[stuck_rows].max(), 0.0)))
        U = U + Jk * (0.6 * depth + 1e-3) / jn
        if not np.all(np.isfinite(U)):
            return U
    return U


# Minimum-effort shoot to a target, subject to inequality constraints:
def lambda_shoot(system, z_target, constraints=(), U0=None, R_weights=None,
                 max_it=1200, ftol=1e-6, outer=25, inner=10, polish=40,
                 end_tol=1e-3):

    # Solver settings, swept across twelve systems and insensitive to all of them:
    reg = 1e-10
    depth = 8
    rho0, rho_max = 10.0, 1e8
    tol_feas = 1e-6
    tol_act = 1e-4
    tol_c = 1e-3
    jac_tol = 0.05
    radius0, radius_max = 0.1, 1e6
    accept_ratio, shrink, grow = 0.1, 0.25, 2.0

    # Only focus on constrained states:
    zt = system.target(z_target)
    m = system.m
    tidx = list(system.tidx)
    n_dof = system.N * system.nu

    # Stacked control weight R_N, and its inverse:
    r_diag = expand_weights(R_weights, system.N, system.nu)
    R_N = 2.0 * r_diag
    R_N_inv = 1.0 / R_N

    # Compile the constraints once and cache them on the system:
    funcs = list(constraints)
    if funcs:
        cache = getattr(system, "_con_cache", None)
        if cache is None:
            cache = system._con_cache = {}
        key = tuple(funcs)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = compile_constraints(system, funcs)
        con = hit
    else:
        con = None

    # Newton shoot for a feasible control sequence, then off any saddle:
    U = newton_shoot(system, zt, U0)
    if con is not None:
        U = _break_symmetry(system, con, U, n_dof)
    if not np.all(np.isfinite(U)):
        system._infeasible = True
        print("[GRACE] WARNING: target does not appear reachable. "
              "Try a longer horizon N, more time, or a closer target.")
        return U

    # Constraint values, and their Jacobian only when it is actually needed:
    def evaluate(V, want_jac=False):
        if con is None:
            return np.zeros(0), np.zeros((0, n_dof))
        return eval_constraints(system, con, np.asarray(V, float).flatten(),
                                want_jac)

    # One multiplier per constraint row:
    h0, _ = evaluate(U)
    mu = np.zeros(h0.size)

    # Penalty scaled to where the cost and constraint gradients are comparable:
    rho = rho0
    if h0.size:
        _, J0 = evaluate(U, want_jac=True)
        g0 = np.linalg.norm(weighted_cost_grad(U, r_diag))
        jn = np.linalg.norm(J0, axis=1)
        jn = jn[jn > 0]
        if jn.size and g0 > 0:
            rho = float(np.clip(g0 / np.median(jn), rho0, rho_max))
    prev_worst = np.inf

    # Shortest step seen, and the iterate that produced it:
    U_best = U.copy()
    best_step = np.inf

    # Best round overall, since a later round may wander away from a good answer:
    U_keep = None
    keep_key = None

    # Augmented-Lagrangian penalty at a control:
    def penalty(V):
        h, _ = evaluate(V)
        if h.size == 0:
            return 0.0, -np.inf
        eta = np.maximum(0.0, mu + rho * h)
        return float(np.sum(eta ** 2 - mu ** 2)) / (2.0 * rho), float(h.max())

    # Outer rounds, one multiplier update each:
    for outer_it in range(outer if con is not None else 1):
        hist_U, hist_F, held = [], [], None
        radius = radius0 * max(float(np.sqrt(max(U @ (r_diag * U), 0.0))), 1.0)
        U_jac = U.copy()
        stale = False
        blend = 1.0
        prev_step = np.inf

        # Inner rounds, minimizing the subproblem at the current multipliers:
        for inner_it in range(inner if con is not None else max_it):
            if not np.all(np.isfinite(U)):
                U = U_best
                break
            e, Co = system.endpoint_jac(U)
            if not np.all(np.isfinite(e)):
                U = U_best
                break
            delta = e - zt

            # Cost gradient, with the active constraint rows folded in:
            g = weighted_cost_grad(U, r_diag)
            G = np.zeros((0, n_dof))
            if con is not None:
                h, _ = evaluate(U)
                eta = np.maximum(0.0, mu + rho * h)

                # A row on its limit counts even once its multiplier has decayed:
                act = np.where((eta > 0.0) | (h > -tol_act))[0]

                # Only the active rows enter the step:
                if act.size:

                    # The Jacobian is stale once the trajectory has drifted from it:
                    drift = np.linalg.norm(U - U_jac)
                    if held is None or stale or \
                            drift > jac_tol * max(np.linalg.norm(U), 1.0):
                        _, held = evaluate(U, want_jac=True)
                        U_jac = U.copy()
                        stale = False
                    g = g + eta[act] @ held[act]
                    G = held[act]

            # Active rows deform the control weight to R_eff = R_N + rho G'G:
            if G.shape[0]:
                n_act = G.shape[0]
                M_act = np.eye(n_act) / rho + (G * R_N_inv) @ G.T \
                    + 1e-12 * np.eye(n_act)

                # Woodbury, so the solve is the size of the active set:
                def apply_R_eff_inv(V, G=G, M_act=M_act):
                    corr = G.T @ safe_solve(M_act, (G * R_N_inv) @ V)
                    return (R_N_inv[:, None] * (V - corr)) if V.ndim > 1 \
                        else R_N_inv * (V - corr)
            else:

                # No active rows, so the weight is R_N itself:
                def apply_R_eff_inv(V):
                    return (R_N_inv[:, None] * V) if V.ndim > 1 else R_N_inv * V

            # Weighted output controllability Gramian and the lambda step:
            R_inv_Co = apply_R_eff_inv(Co.T)
            W_R = Co @ R_inv_Co + reg * np.eye(m)
            R_inv_grad = apply_R_eff_inv(g)
            lam = safe_solve(W_R, Co @ R_inv_grad - delta)
            dU = -(R_inv_grad - R_inv_Co @ lam)
            step_norm = np.linalg.norm(dU)
            if not np.isfinite(step_norm):
                U = U_best
                break

            # An exact penalty needs a weight above the endpoint multiplier:
            sigma = 2.0 * float(np.max(np.abs(lam))) + 1.0

            # Track the shortest step, and stop once it is small enough:
            if step_norm < best_step:
                best_step = step_norm
                U_best = U.copy()
            if step_norm < ftol * max(np.linalg.norm(U), 1.0):
                break

            # Unconstrained, the step is a fixed-point map that barely contracts:
            if con is None:

                # A growing step means the extrapolation has failed, so restart it:
                if step_norm > prev_step:
                    hist_U.clear()
                    hist_F.clear()
                    blend = max(blend * 0.5, 1e-3)
                else:
                    blend = min(blend * 1.2, 1.0)
                prev_step = step_norm

                # Keep the last few residuals:
                hist_U.append(U.copy())
                hist_F.append(dU.copy())
                if len(hist_U) > depth:
                    hist_U.pop(0)
                    hist_F.pop(0)

                # Plain step on the first pass, mixed step after:
                k = len(hist_U)
                if k == 1:
                    U_new = U + blend * dU
                else:
                    dF = np.column_stack(
                        [hist_F[i + 1] - hist_F[i] for i in range(k - 1)])
                    dU_hist = np.column_stack(
                        [hist_U[i + 1] - hist_U[i] for i in range(k - 1)])

                    # Anderson: subtract the part of the residual the history explains:
                    try:
                        gamma = np.linalg.lstsq(dF, dU, rcond=1e-10)[0]
                        U_new = U + blend * dU - (dU_hist + blend * dF) @ gamma
                    except np.linalg.LinAlgError:
                        U_new = U + blend * dU

                # Fall back to the plain step if the mixed one is wild:
                if not np.all(np.isfinite(U_new)) or \
                        np.linalg.norm(U_new - U) > 50.0 * max(step_norm, 1e-12):
                    hist_U.clear()
                    hist_F.clear()
                    blend = max(blend * 0.5, 1e-3)
                    U_new = U + blend * dU

                # Optional trace of the damping schedule:
                if getattr(system, "_count_ls", False):
                    print("[shoot] it %4d  blend %8.3g  step %9.3e"
                          % (inner_it, blend, step_norm))
                U = U_new
                continue

            # Merit at the current iterate:
            pen_now, _ = penalty(U)
            merit = weighted_cost(U, r_diag) + pen_now \
                + sigma * np.linalg.norm(delta)

            # Clip the step to the radius, in the metric the cost is measured in:
            p = dU
            p_norm = float(np.sqrt(max(p @ (r_diag * p), 0.0)))
            if p_norm > radius:
                p = p * (radius / p_norm)
            predicted = -float(g @ p)

            # One evaluation, rather than a search over step lengths:
            U_try = U + p
            Z_try = system.rollout(U_try)
            moved = False
            if np.all(np.isfinite(Z_try)):
                pen_try, _ = penalty(U_try)
                achieved = merit - (weighted_cost(U_try, r_diag) + pen_try
                                    + sigma * np.linalg.norm(Z_try[-1][tidx] - zt))

                # How much of the promised improvement the step delivered:
                ratio = achieved / max(abs(predicted), 1e-16)

                # Accept a decent share, and grow the radius when the model holds:
                if ratio > accept_ratio and achieved > 0.0:
                    U = U_try
                    moved = True
                    if ratio > 0.75 and p_norm >= radius:
                        radius = min(radius * grow, radius_max)

                # A poor step means the model does not hold this far out:
                elif ratio < 0.25:
                    radius = max(radius * shrink, 1e-10)
            else:
                radius = max(radius * shrink, 1e-10)

            # A rejected step just means try shorter, or rebuild the Jacobian:
            if not moved:
                if radius > 1e-9:
                    continue
                if not stale:
                    stale = True
                    radius = radius0 * max(
                        float(np.sqrt(max(U @ (r_diag * U), 0.0))), 1.0)
                    continue
                break

        # Unconstrained solves are one pass, so the outer loop ends here:
        if con is None:
            break

        # Multiplier update, and the complementarity it has to satisfy:
        h, _ = evaluate(U)
        worst = float(h.max()) if h.size else -np.inf
        mu = np.maximum(0.0, mu + rho * h)
        slack = float(np.max(mu * np.maximum(0.0, -h))) if h.size else 0.0

        # Rank this round by feasibility first, then cost, and keep the best:
        end_now = np.linalg.norm(system.endpoint(U) - zt)
        key = (max(worst, 0.0) + max(end_now - 1e-6, 0.0),
               weighted_cost(U, r_diag))
        if np.all(np.isfinite(U)) and (keep_key is None or key < keep_key):
            keep_key = key
            U_keep = U.copy()

        # One update must happen first, or complementarity was never tested:
        if outer_it > 0 and worst < tol_feas and slack < tol_c:
            break

        # Slack grows with rho, so a stalled inner solve is the other exit:
        if outer_it > 0 and worst < tol_feas and inner_it <= 1:
            break

        # Tighten only when the violation did not fall enough, and cap it:
        if worst > 0.5 * prev_worst:
            rho = min(rho * 4.0, rho_max)
        prev_worst = worst

    # Fall back to the best round rather than the last one:
    if con is not None and U_keep is not None:
        U = U_keep

    # Unconstrained, a plain feasibility shoot restores the endpoint:
    if con is None:
        U = newton_shoot(system, zt, U)

    # Constrained, it cannot: at convergence the trajectory sits on the active
    # rows, so closing the endpoint alone pushes straight back through them:
    else:
        for _ in range(25):
            e, Co = system.endpoint_jac(U)
            r = e - zt
            h, Jh = evaluate(U, want_jac=True)

            # Endpoint residual and active violations, driven down together:
            act = np.where(h > -tol_feas)[0] if h.size else np.array([], int)
            resid = np.concatenate([r, np.maximum(h[act], 0.0)]) if act.size else r
            if np.linalg.norm(resid) < 1e-11:
                break

            # Active rows stacked onto the endpoint Jacobian:
            A = np.vstack([Co, Jh[act]]) if act.size else Co

            # Least-norm correction that cancels both at the linearization:
            step = -A.T @ safe_solve(A @ A.T + reg * np.eye(A.shape[0]), resid)
            moved = False
            for a in [1.0, 0.5, 0.25, 0.1, 0.05]:
                U_try = U + a * step
                e_try = system.endpoint(U_try)
                if not np.all(np.isfinite(e_try)):
                    continue
                h_try, _ = evaluate(U_try)
                r_try = np.concatenate(
                    [e_try - zt, np.maximum(h_try[act], 0.0)]) if act.size \
                    else e_try - zt
                if np.linalg.norm(r_try) < np.linalg.norm(resid):
                    U = U_try
                    moved = True
                    break

            # No step reduced the residual, so there is nothing left to gain:
            if not moved:
                break

    # Polish: the augmented Lagrangian reaches feasibility but never then
    # reduces cost along the constraint surface, which this does:
    if con is not None:
        for _ in range(polish):
            h, Jh = evaluate(U, want_jac=True)
            _, Co = system.endpoint_jac(U)
            act = np.where(h > -tol_act)[0] if h.size else np.array([], int)
            A = np.vstack([Co, Jh[act]]) if act.size else Co

            # Cost gradient, with anything that moves the endpoint or the active
            # rows projected out, leaving a slide along the surface:
            g = weighted_cost_grad(U, r_diag)
            R_inv_g = g / R_N
            R_inv_A = (A / R_N).T
            gram = A @ R_inv_A + reg * (np.trace(A @ R_inv_A) / A.shape[0] + 1.0) \
                * np.eye(A.shape[0])
            p = -(R_inv_g - R_inv_A @ safe_solve(gram, A @ R_inv_g))
            p_norm = float(np.sqrt(max(p @ (R_N * p), 0.0)))
            if p_norm < 1e-10 * max(float(np.sqrt(max(U @ (R_N * U), 0.0))), 1.0):
                break

            # Where the solve currently stands, so a step cannot give any of it up:
            cost_now = weighted_cost(U, r_diag)
            end_now = np.linalg.norm(system.endpoint(U) - zt)
            worst_now = float(h.max()) if h.size else -np.inf
            moved = False

            # The useful move is a short slide, so the ladder reaches far smaller
            # steps than the main loop needs:
            for a in [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]:
                U_try = U + a * p
                Z_try = system.rollout(U_try)
                if not np.all(np.isfinite(Z_try)):
                    continue
                h_try, _ = evaluate(U_try)
                if float(h_try.max()) > max(worst_now, tol_feas):
                    continue

                # The projection cancels endpoint motion only to first order, so
                # allow a small drift here and restore it exactly afterwards:
                if np.linalg.norm(Z_try[-1][tidx] - zt) > max(end_now * 10.0, end_tol):
                    continue
                if weighted_cost(U_try, r_diag) < cost_now:
                    U = U_try
                    moved = True
                    break
            if not moved:
                break

        # Put the endpoint back, with the active rows in the projection:
        for _ in range(15):
            e, Co = system.endpoint_jac(U)
            r = e - zt
            h, Jh = evaluate(U, want_jac=True)
            act = np.where(h > -tol_feas)[0] if h.size else np.array([], int)
            resid = np.concatenate([r, np.maximum(h[act], 0.0)]) if act.size else r
            if np.linalg.norm(resid) < 1e-11:
                break
            A = np.vstack([Co, Jh[act]]) if act.size else Co
            step = -A.T @ safe_solve(A @ A.T + reg * np.eye(A.shape[0]), resid)
            ok_step = False
            for a in [1.0, 0.5, 0.25, 0.1]:
                U_try = U + a * step
                e_try = system.endpoint(U_try)
                if not np.all(np.isfinite(e_try)):
                    continue
                h_try, _ = evaluate(U_try)
                r_try = np.concatenate(
                    [e_try - zt, np.maximum(h_try[act], 0.0)]) if act.size \
                    else e_try - zt
                if np.linalg.norm(r_try) < np.linalg.norm(resid):
                    U = U_try
                    ok_step = True
                    break
            if not ok_step:
                break

    # Flag if the endpoint was missed or any constraint is still violated:
    end_err = np.linalg.norm(system.endpoint(U) - zt)
    h, _ = evaluate(U)
    worst = float(h.max()) if h.size else -np.inf
    system._infeasible = bool(end_err > 1e-2 or worst > 1e-2)
    if system._infeasible:
        print("[GRACE] WARNING: request not fully satisfied (endpoint error %.2e, "
              "worst constraint violation %.2e). Try a longer horizon N, more "
              "time, or looser limits." % (end_err, max(worst, 0.0)))

    # Return control:
    return U