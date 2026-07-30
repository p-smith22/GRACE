# Import packages:
import numpy as np

# Trace a collision-free path from start to goal through a potential flow field:
def potential_path(start, goal, OBS, R, step=0.03, maxsteps=6000):

    # Initialize path:
    p = np.array(start, float)
    path = [p.copy()]

    # March along the local flow direction until the goal is reached:
    for _ in range(maxsteps):

        # Sink term aimed at the goal, which terminates the march when close:
        to_goal = goal - p
        d_goal = np.linalg.norm(to_goal)
        if d_goal < step * 2:
            break
        v = to_goal / d_goal

        # Superpose the source and vortex of every obstacle within influence:
        for o in OBS:
            d = p - o
            dist = np.linalg.norm(d)
            influence = R * 3.5
            if dist < influence:

                # Outward normal, floored so the field stays finite at center:
                n = d / max(dist, 1e-6)
                strength = (R / max(dist, 0.25)) ** 2

                # Source pushes away from the obstacle:
                v = v + n * strength * 1.2

                # Vortex circulates, signed toward the side the goal is on:
                perp = np.array([-n[1], n[0]])
                if perp @ (goal - p) < 0:
                    perp = -perp
                v = v + perp * strength * 1.6

        # A vanishing field means the terms have cancelled, so stop marching:
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            break

        # Advance a fixed arc length along the flow direction:
        p = p + step * v / v_norm
        path.append(p.copy())

    # Close the path on the goal exactly, since the march stops just short:
    path.append(np.array(goal, float))
    return np.array(path)


# Resample a traced path onto the N+1 trajectory nodes by arc length:
def _reference_nodes(path, N):

    # Accumulate arc length for node discretization:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    arc = arc / max(arc[-1], 1e-9)

    # One reference position per node:
    idx = [min(np.searchsorted(arc, k / N), len(path) - 1) for k in range(N + 1)]
    return np.array([path[i] for i in idx])

# Evaluate the one-step Jacobian tape along a trajectory:
def _jacobian_tape(s, Z, U):

    # Prefer the batched call, which evaluates the whole tape in one shot:
    if getattr(s, "step_jac_all", None) is not None:
        return s.step_jac_all(Z, U)

    # Fall back to per-node calls if the system was built without the map:
    nu = s.nu
    A = [np.asarray(s.step_jac(Z[k], U[k * nu:(k + 1) * nu])[0]) for k in range(s.N)]
    B = [np.asarray(s.step_jac(Z[k], U[k * nu:(k + 1) * nu])[1]) for k in range(s.N)]
    return A, B


# Build the terminal value function that pulls the endpoint onto the target:
def _terminal_value(s, Z, z_target, tidx, w_end):

    # Quadratic penalty on the targeted states only:
    P = np.zeros((s.nx, s.nx))
    v = np.zeros(s.nx)
    for j, t in enumerate(tidx):
        P[t, t] = w_end
        v[t] = w_end * (Z[-1, t] - z_target[j])
    return P, v


# Run the backward Riccati recursion and return the feedforward and feedback:
def _backward_pass(s, U, A, B, Qk, qk, P, v, R_eff):

    # Standard discrete recursion:
    N, nu = s.N, s.nu
    K = [None] * N
    k_ff = [None] * N
    for k in range(N - 1, -1, -1):
        Ak, Bk = A[k], B[k]
        Quu = R_eff + Bk.T @ P @ Bk + 1e-2 * np.eye(nu)
        Qux = Bk.T @ P @ Ak
        qu = Bk.T @ v + R_eff @ U[k * nu:(k + 1) * nu]

        # Feedback and feedforward for this node:
        K[k] = np.linalg.solve(Quu, Qux)
        k_ff[k] = np.linalg.solve(Quu, qu)

        # Propagate the value function one step earlier:
        P = Qk[k] + Ak.T @ P @ Ak - Qux.T @ K[k]
        v = qk[k] + Ak.T @ v - Qux.T @ k_ff[k]
    return K, k_ff


# Line search over step size, rolling the closed loop out and testing accept:
def _forward_pass(s, U, Z, K, k_ff, clip, accept):

    # Initialize:
    N, nu = s.N, s.nu
    z0 = np.asarray(s.z0, float)
    for a in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
        U_new = U.copy()
        z = z0.copy()
        finite = True

        # Roll forward, correcting for the deviation from the nominal:
        for k in range(N):
            du = -a * k_ff[k] - K[k] @ (z - Z[k])
            U_new[k * nu:(k + 1) * nu] = U[k * nu:(k + 1) * nu] + du
            U_new = clip(U_new)
            z = s.step_np(z, U_new[k * nu:(k + 1) * nu])
            if not np.all(np.isfinite(z)):
                finite = False
                break

        # Return the first candidate that is finite and passes the test:
        if finite and accept(U_new):
            return U_new

    # No step size was acceptable, which the caller reads as convergence:
    return None


# Build a control clipping function from optional per-node bounds:
def _clipper(N, u_lo, u_hi):

    # Bounds are given per control and tiled across the horizon:
    if u_lo is None and u_hi is None:
        return lambda U: U
    lo = None if u_lo is None else np.tile(u_lo, N)
    hi = None if u_hi is None else np.tile(u_hi, N)
    return lambda U: np.clip(U, lo, hi)


# Build the effort cost and its gradient from an optional control weighting:
def _effort(N, nu, R_weights):

    # Define objective helper function:
    if R_weights is None:
        return (lambda U: float(U @ U)), (lambda U: 2.0 * U)
    W = np.asarray(R_weights, float)

    # A vector is read as the diagonal, which is the common case:
    if W.ndim == 1:
        if W.size != nu:
            raise ValueError("R_weights vector must have nu=%d entries, got %d" % (nu, W.size))
        w = np.tile(W, N)
        return (lambda U: float(U @ (w * U))), (lambda U: 2.0 * (w * U))

    # A full matrix is applied blockwise at every node:
    if W.shape != (nu, nu):
        raise ValueError("R_weights matrix must be (%d, %d), got %s" % (nu, nu, W.shape))

    def cost(U):
        Um = U.reshape(N, nu)
        return float(np.sum(Um * (Um @ W.T)))

    def grad(U):
        Um = U.reshape(N, nu)
        return (Um @ (W + W.T)).flatten()

    return cost, grad

# Drive the trajectory onto a reference path while reaching the endpoint:
def _track_path(s, z_target, pos_idx, tidx, ref, clip, iters=60):

    # Unpack:
    N, nu, nx = s.N, s.nu, s.nx
    U = np.zeros(N * nu)
    R_eff = np.eye(nu) * 1e-3
    w_end = 5e4
    w_track = 3.0

    # Endpoint error on the targeted states only:
    def endpoint_error(Uv):
        z = s.rollout(Uv)[-1]
        return np.array([z[t] - z_target[j] for j, t in enumerate(tidx)])

    # Squared deviation of the position history from the reference path:
    def path_deviation(Uv):
        p = s.rollout(Uv)[:, pos_idx]
        return np.sum((p - ref[:len(p)]) ** 2)

    # Merit combines path tracking, endpoint, and a token effort term:
    def merit(Uv):
        e = endpoint_error(Uv)
        return w_track * path_deviation(Uv) + w_end * float(e @ e) + float(Uv @ Uv) * 1e-3

    # Iterate the linear-quadratic subproblem to convergence:
    for _ in range(iters):
        Z = s.rollout(U)

        # Running cost is the path tracking penalty at every node:
        Qk = [np.zeros((nx, nx)) for _ in range(N + 1)]
        qk = [np.zeros(nx) for _ in range(N + 1)]
        for k in range(N + 1):
            for a in range(2):
                Qk[k][pos_idx[a], pos_idx[a]] += 2 * w_track
                qk[k][pos_idx[a]] += 2 * w_track * (Z[k, pos_idx[a]] - ref[k, a])

        # Solve the subproblem and take the first step that lowers merit:
        A, B = _jacobian_tape(s, Z, U)
        P, v = _terminal_value(s, Z, z_target, tidx, w_end)
        K, k_ff = _backward_pass(s, U, A, B, Qk, qk, P, v, R_eff)
        m0 = merit(U)
        U_new = _forward_pass(s, U, Z, K, k_ff, clip, lambda Uv: merit(Uv) < m0 - 1e-9)
        if U_new is None:
            break
        U = U_new
    return U

# Route and track repeatedly, inflating the routing radius by the penetration:
def _route_and_track(s, z_target, obstacles, R, pos_idx, tidx, clip, verbose=False):

    # Set goal:
    goal = z_target[[list(tidx).index(pos_idx[0]), list(tidx).index(pos_idx[1])]]

    # Worst-case clearance across all obstacles over the whole trajectory:
    def clearance(Uv):
        p = s.rollout(Uv)[:, pos_idx]
        return min((np.sum((p - o) ** 2, axis=1) ** 0.5).min() for o in obstacles)

    # Endpoint error norm on the targeted states:
    def endpoint_error(Uv):
        z = s.rollout(Uv)[-1]
        return np.linalg.norm([z[t] - z_target[j] for j, t in enumerate(tidx)])

    # Inflate until the tracked trajectory clears the true radius:
    R_route = R
    U = None
    for attempt in range(8):
        path = potential_path(s.z0[pos_idx], goal, obstacles, R_route)
        ref = _reference_nodes(path, s.N)
        U = _track_path(s, z_target, pos_idx, tidx, ref, clip)
        c = clearance(U)
        e = endpoint_error(U)
        if verbose:
            print("  attempt %d R_route %.2f -> clearance %.3f penetration %.3f endpoint %.3f"
                  % (attempt, R_route, c, max(0.0, R - c), e))

        # Done once the trajectory is clean and the endpoint is reached:
        if c >= R and e < 0.05:
            return U, R_route, c, e

        # Otherwise inflate by the penetration plus a small increment:
        R_route += max(R - c, 0.0) + 0.15

    # Return the best attempt, which the entry point will flag as infeasible:
    return U, R_route, clearance(U), endpoint_error(U)

# Hold the clearance won by tracking while driving every targeted state home:
def _close_endpoint(s, z_target, obstacles, R, pos_idx, tidx, U, clearance_won, clip,
                    iters=50):

    # Unpack:
    N, nu, nx = s.N, s.nu, s.nx
    R_eff = np.eye(nu) * 1e-3
    w_end = 1e5

    # Keep the margin the tracking stage actually achieved:
    margin = max(0.15, (clearance_won - R) * 0.5)

    # Worst-case clearance, used as the hard acceptance veto:
    def clearance(Uv):
        p = s.rollout(Uv)[:, pos_idx]
        return min((np.sum((p - o) ** 2, axis=1) ** 0.5).min() for o in obstacles)

    # Squared penetration summed over nodes and obstacles:
    def penetration(Uv):
        p = s.rollout(Uv)[:, pos_idx]
        total = 0.0
        for o in obstacles:
            d = np.sqrt(np.sum((p - o) ** 2, axis=1))
            total += np.sum(np.maximum(0.0, R - d) ** 2)
        return total

    # Endpoint error on the targeted states:
    def endpoint_error(Uv):
        z = s.rollout(Uv)[-1]
        return np.array([z[t] - z_target[j] for j, t in enumerate(tidx)])

    # Merit trades a token effort term against penetration and endpoint:
    def merit(Uv):
        e = endpoint_error(Uv)
        return float(Uv @ Uv) * 1e-3 + 1e4 * penetration(Uv) + w_end * float(e @ e)

    # Iterate the linear-quadratic subproblem to convergence:
    for _ in range(iters):
        Z = s.rollout(U)
        p = Z[:, pos_idx]

        # Running cost:
        Qk = [np.zeros((nx, nx)) for _ in range(N + 1)]
        qk = [np.zeros(nx) for _ in range(N + 1)]
        for k in range(N + 1):
            for o in obstacles:
                d = p[k] - o
                dist = np.sqrt(np.sum(d ** 2) + 1e-9)
                violation = (R + margin) - dist
                if violation > 0:

                    # Gradient and Gauss-Newton Hessian of the penalty, taken
                    # along the outward normal:
                    n = d / dist
                    rho = 120.0
                    qk[k][pos_idx[0]] += -2 * rho * violation * n[0]
                    qk[k][pos_idx[1]] += -2 * rho * violation * n[1]
                    Q_pos = 2 * rho * np.outer(n, n)
                    for a in range(2):
                        for b in range(2):
                            Qk[k][pos_idx[a], pos_idx[b]] += Q_pos[a, b]

        # Accept only steps that lower merit and are still strictly clean:
        A, B = _jacobian_tape(s, Z, U)
        P, v = _terminal_value(s, Z, z_target, tidx, w_end)
        K, k_ff = _backward_pass(s, U, A, B, Qk, qk, P, v, R_eff)
        m0 = merit(U)

        def accept(Uv):
            return clearance(Uv) >= R and merit(Uv) < m0 - 1e-9

        U_new = _forward_pass(s, U, Z, K, k_ff, clip, accept)
        if U_new is None:
            break
        U = U_new
    return U

# Reduce effort in the null space of the active constraints until stationary:
def _polish(s, z_target, obstacles, R, pos_idx, tidx, U, clip, iters=250, reg=1e-8,
            effort_cost=None, effort_grad=None):

    # Unpack constrained states:
    m = len(tidx)

    # Effort cost definition:
    if effort_cost is None:
        effort_cost = lambda Uv: float(Uv @ Uv)
    if effort_grad is None:
        effort_grad = lambda Uv: 2.0 * Uv

    # Worst-case clearance across all obstacles:
    def clearance(Uv):
        p = s.rollout(Uv)[:, pos_idx]
        return min((np.sum((p - o) ** 2, axis=1) ** 0.5).min() for o in obstacles)

    # Endpoint error on the targeted states:
    def endpoint_error(Uv):
        return s.endpoint(Uv) - z_target

    # Restore the endpoint by least-norm correction, vetoing penetration:
    def reproject(Uv, iters=12):
        for _ in range(iters):
            r = endpoint_error(Uv)
            if np.linalg.norm(r) < 1e-7:
                break

            # Least-norm step that cancels the endpoint error:
            _, J = s.endpoint_jac(Uv)
            full = -J.T @ np.linalg.solve(J @ J.T + reg * np.eye(m), r)

            # Shorten until the correction does not enter an obstacle:
            accepted = False
            for a in [1.0, 0.5, 0.25, 0.1, 0.05]:
                U_try = clip(Uv + a * full)
                if clearance(U_try) >= R - 1e-9:
                    Uv = U_try
                    accepted = True
                    break
            if not accepted:
                break
        return Uv

    # Descend the residual gradient with an adaptive trust region:
    cost = effort_cost(U)
    trust = 0.3
    stalls = 0
    for _ in range(iters):

        # Active set is the endpoint rows plus the obstacle rows:
        _, C_end = s.endpoint_jac(U)
        p = s.rollout(U)[:, pos_idx]
        J_pos = np.array(s.pos_jac(U))
        rows = [C_end]
        for o in obstacles:
            dist = np.sqrt(np.sum((p - o) ** 2, axis=1))

            # Only nodes essentially on the boundary count as active:
            for k in np.where(dist < R + 1e-3)[0]:
                n = (p[k] - o) / max(dist[k], 1e-9)
                rows.append((n @ J_pos[k]).reshape(1, -1))
        A = np.vstack(rows)
        W = A @ A.T + reg * np.eye(A.shape[0])

        # Project the pure effort gradient onto the constraint null space:
        g = effort_grad(U)
        residual = g - A.T @ np.linalg.solve(W, A @ g)

        # A vanishing residual relative to the gradient is stationarity:
        if np.linalg.norm(residual) / max(np.linalg.norm(g), 1e-9) < 1e-4:
            break

        # Step along the normalized descent direction, then restore endpoint:
        d = -residual / np.linalg.norm(residual)
        U_try = reproject(clip(U + trust * np.linalg.norm(U) * d))

        # Accept only if still clean, endpoint held, and cost strictly lower:
        if (clearance(U_try) >= R - 1e-9
                and np.linalg.norm(endpoint_error(U_try)) < 1e-4
                and effort_cost(U_try) < cost - 1e-12):
            U = U_try
            cost = effort_cost(U)
            trust = min(trust * 1.5, 1.0)
            stalls = 0
        else:
            trust *= 0.5
            stalls += 1

            # Restart if TR stalls:
            if trust < 1e-7:
                if stalls > 5:
                    break
                trust = 0.3
    return U

# Minimum-effort shoot to a target while keeping clear of circular obstacles:
def lambda_obstacle(system, z_target, obstacles, R, pos_idx=(0, 1), max_it=250, reg=1e-8,
                    u_lo=None, u_hi=None, R_weights=None, verbose=False):

    # Must have position defintiion of where to avoid obstacle:
    pi = list(pos_idx)
    tidx = list(system.tidx)
    missing = [i for i in pi if i not in tidx]
    if missing:
        raise ValueError(
            "obstacle avoidance needs pos_idx %s to be included in the system's target_idx %s "
            "(missing %s) -- rebuild the system with those position states in target_idx, "
            "or pass the correct pos_idx." % (tuple(pi), tuple(tidx), tuple(missing)))

    # Require position index:
    if getattr(system, "pos_jac", None) is None:
        raise ValueError(
            "obstacle avoidance needs a position Jacobian -- rebuild the system with "
            "pos_idx=%s." % (tuple(pi),))

    # Geometry, bounds, and the effort objective:
    OBS = [np.asarray(o, float) for o in obstacles]
    zt = system.target(z_target)
    clip = _clipper(system.N, u_lo, u_hi)
    effort_cost, effort_grad = _effort(system.N, system.nu, R_weights)

    # Find feasible trajectory and polish:
    U, _R_route, clearance_won, _err = _route_and_track(
        system, zt, OBS, R, pi, tidx, clip, verbose)
    U = _close_endpoint(system, zt, OBS, R, pi, tidx, U, clearance_won, clip)
    U = _polish(system, zt, OBS, R, pi, tidx, U, clip, iters=max_it, reg=reg,
                effort_cost=effort_cost, effort_grad=effort_grad)

    # Report results:
    Z = system.rollout(U)
    clr = min((np.sum((Z[:, pi] - o) ** 2, axis=1) ** 0.5).min() for o in OBS)
    err = np.linalg.norm(np.array([Z[-1, t] - zt[j] for j, t in enumerate(tidx)]))
    system._obstacle_infeasible = bool(clr < R - 1e-2 or err > 5e-2)
    if system._obstacle_infeasible:
        print("[grace] warning: obstacle request appears infeasible for this horizon "
              "(clearance %.2f vs R %.2f, endpoint error %.2e). Try a longer horizon N, "
              "more time, or an easier obstacle." % (clr, R, err))

    # Return optimal control:
    return U
