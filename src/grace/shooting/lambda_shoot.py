# Import packages:
import numpy as np
from scipy.optimize import root
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

    # Every constraint goes into one block. A mapped call costs the same fixed
    # overhead whether it carries one row or ten, and it is made once per node
    # sweep, so a separate map per constraint pays that overhead once per
    # constraint for no extra work done. The union of the states and controls
    # any of them touches is still a short list, so stacking them differentiates
    # only a few structural zeros more than keeping them apart:
    exprs = [f(z, u) for f in funcs]
    expr = ca.vertcat(*exprs)
    nrow = int(expr.numel())

    # Only the states and controls the set touches are differentiated:
    s_dep = [i for i, b in enumerate(ca.which_depends(expr, z, 1, False)) if b]
    u_dep = [i for i, b in enumerate(ca.which_depends(expr, u, 1, False)) if b]

    # Local gradients, before any chaining through the trajectory:
    Jz = ca.jacobian(expr, z)
    Ju = ca.jacobian(expr, u)

    # Value and gradient separately, since values are asked for far more often:
    g_val = ca.Function("g", [z, u], [expr])
    g_jac = ca.Function("gj", [z, u],
                        [Jz[:, s_dep] if s_dep else ca.MX.zeros(nrow, 0),
                         Ju[:, u_dep] if u_dep else ca.MX.zeros(nrow, 0)])

    # Map over every node and store:
    f_val = ca.Function.map(g_val, K)
    f_jac = ca.Function.map(g_jac, K)
    blocks = [(s_dep, u_dep, f_val, f_jac, nrow)]

    # Union of the states any constraint needs, so one row Jacobian serves all:
    states = tuple(sorted(s_dep))
    return blocks, states, K

# Evaluate every constraint at a control sequence:
def eval_constraints(system, con, U, want_jac=False, Z=None):

    # Unpack, reusing a rollout the caller already has:
    blocks, states, K = con
    N, nu = system.N, system.nu
    if Z is None:
        Z = np.asarray(system.rollout(U), float)
    else:
        Z = np.asarray(Z, float)

    # The terminal node has no control of its own, so it reuses the last one:
    idx = [min(k, N - 1) for k in range(K)]
    Un = np.stack([U[i * nu:(i + 1) * nu] for i in idx], axis=1)
    JZ = system.row_jac(U, states) if (want_jac and states) else None

    # Values and gradient rows, block by block:
    h_all = []
    J_all = []

    # Loop through each block:
    for s_dep, u_dep, f_val, f_jac, nrow in blocks:

        # A mapped call returns one column per node, so the rows come out
        # constraint-major and stay in the order the caller wrote them:
        h_all.append(np.array(f_val(Z.T, Un)).reshape(nrow, K).flatten())
        if not want_jac:
            continue

        # Local gradients at every node, one column block per node:
        gz, gu = f_jac(Z.T, Un)

        # Chain the state part with dz/dU:
        rows = np.zeros((nrow, K, N * nu))
        if s_dep:
            cols = [states.index(i) for i in s_dep]
            gz = np.array(gz).reshape(nrow, K, len(s_dep))
            rows += np.einsum("cki,kid->ckd", gz, JZ[:, cols, :])

        # A control affects only its own node, so its gradient is placed directly:
        if u_dep:
            gu = np.array(gu).reshape(nrow, K, len(u_dep))
            for k in range(K):
                base = idx[k] * nu
                for c, j in enumerate(u_dep):
                    rows[:, k, base + j] += gu[:, k, c]

        # Store this block's rows, flattened to match the values:
        J_all.append(rows.reshape(nrow * K, N * nu))

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

# Apply an inverse-Hessian metric, diagonal or block diagonal, to a vector or matrix:
def _apply_metric(dp, V, nu):

    # A flat dp is a diagonal metric, so it multiplies elementwise:
    if dp.ndim == 1:
        return dp[:, None] * V if V.ndim > 1 else dp * V

    # A block dp couples the controls within a node but not across nodes:
    W = V if V.ndim > 1 else V[:, None]
    Wb = W.reshape(dp.shape[0], nu, W.shape[1])
    out = np.einsum("kij,kjc->kic", dp, Wb).reshape(-1, W.shape[1])
    return out if V.ndim > 1 else out[:, 0]

# Minimum-effort step from per-node linearizations, without the composed Jacobian:
def _recursive_step(system, U, zt, R_N_inv, reg, m, tidx):

    # One linearization per node, differentiating a single RK4 step N times
    # rather than differentiating through the whole composed rollout:
    Z, A, B = system.node_jac(U)
    N, nu, nx = system.N, system.nu, system.nx
    e = np.asarray(Z)[-1][tidx]
    delta = e - zt

    # Backward sweep. P carries the state transition from node k+1 to the end,
    # so P @ B_k is the k-th block of the endpoint Jacobian and the sum below is
    # the weighted output controllability Gramian, formed without ever storing
    # that Jacobian:
    W = np.zeros((m, m))
    CoV = np.zeros(m)
    P = np.eye(m)
    for k in range(N - 1, -1, -1):
        Bk = B[:, k * nu:(k + 1) * nu][tidx]
        Rk = R_N_inv[k * nu:(k + 1) * nu]
        M = P @ Bk
        W += (M * Rk) @ M.T
        CoV += M @ (Rk * U[k * nu:(k + 1) * nu])
        P = P @ A[:, k * nx:(k + 1) * nx][np.ix_(tidx, tidx)]

    # Endpoint multiplier, the terminal costate of the minimum-effort problem:
    lam = safe_solve(W + reg * np.eye(m), CoV - delta)

    # Adjoint vector sweep for Co^T lam, one matrix-vector product per node:
    dU = np.empty(N * nu)
    p = lam.copy()
    for k in range(N - 1, -1, -1):
        Bk = B[:, k * nu:(k + 1) * nu][tidx]
        Rk = R_N_inv[k * nu:(k + 1) * nu]
        dU[k * nu:(k + 1) * nu] = -(U[k * nu:(k + 1) * nu] - Rk * (Bk.T @ p))
        p = A[:, k * nx:(k + 1) * nx][np.ix_(tidx, tidx)].T @ p
    return e, delta, lam, dU

# Minimum-effort shoot to a target, subject to inequality constraints:
def lambda_shoot(system, z_target, constraints=(), U0=None, R_weights=None,
                 max_it=1200, ftol=1e-6, outer=25, inner=10, polish=40,
                 end_tol=1e-3, nx_recursive=16, cost=None):

    # cost = (f, dinv[, dprime[, grad]]). dinv inverts the cost gradient, dprime
    # is M = (grad^2 f)^-1 as a flat diagonal or an (N,nu,nu) block diagonal,
    # and grad is the cost gradient itself. Unconstrained solves take the exact
    # dual path and need only (f, dinv); inequality rows need dprime and grad,
    # because the active-set metric and the merit test both read the cost
    # gradient at iterates that damping has moved off the dual map:
    f_cost = f_dinv = f_dprime = f_grad = None
    if cost is not None:
        f_cost, f_dinv = cost[0], cost[1]
        f_dprime = cost[2] if len(cost) > 2 else None
        f_grad = cost[3] if len(cost) > 3 else None
        if constraints and (f_dprime is None or f_grad is None):
            raise ValueError(
                "a general cost combined with inequality constraints needs "
                "cost = (f, dinv, dprime, grad): dprime supplies the metric "
                "the active rows deform, and grad is read at damped iterates "
                "where U is no longer dinv(s)")
        if R_weights is not None:
            raise ValueError("R_weights and cost both set the control cost; "
                             "put the weights inside the cost")

    # Solver settings, swept across twelve systems and insensitive to all of them:
    reg = 1e-10
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

    # Recursion pays off at large nx, but assumes a diagonal quadratic weight:
    use_rec = (con is None and cost is None and m >= nx_recursive
               and hasattr(system, "node_jac"))

    # What the merit test charges for control:
    def cost_of(V):
        return f_cost(V) if cost is not None else weighted_cost(V, r_diag)

    # Cost gradient, which is what the active rows are folded into:
    def cost_grad_of(V):
        if cost is None:
            return weighted_cost_grad(V, r_diag)
        return np.asarray(f_grad(V), float).flatten()

    # Metric the step is taken in: R_N^-1 for a quadratic cost, the inverse cost
    # Hessian M(s) for a general one. Everything downstream touches the metric
    # only through this application, so the Woodbury update is the same formula
    # in both cases -- and M appears where M^-1 never does, so a cost with a
    # singular Hessian is still admissible:
    def metric_at(V):
        if cost is None:
            return lambda X: (R_N_inv[:, None] * X) if X.ndim > 1 else R_N_inv * X
        dp = np.asarray(f_dprime(cost_grad_of(V)))
        return lambda X: _apply_metric(dp, X, system.nu)

    # Trust-region length. The quadratic path measures steps in the weight the
    # cost is written in; a general cost has no such fixed weight, so it is
    # measured plainly:
    def tr_norm(V):
        if cost is not None:
            return float(np.linalg.norm(V))
        return float(np.sqrt(max(V @ (r_diag * V), 0.0)))

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
    def evaluate(V, want_jac=False, Z=None):
        if con is None:
            return np.zeros(0), np.zeros((0, n_dof))
        return eval_constraints(system, con, np.asarray(V, float).flatten(),
                                want_jac, Z)

    # One multiplier per constraint row:
    h0, _ = evaluate(U)
    mu = np.zeros(h0.size)

    # Penalty scaled to where the cost and constraint gradients are comparable:
    rho = rho0
    if h0.size:
        _, J0 = evaluate(U, want_jac=True)
        g0 = np.linalg.norm(cost_grad_of(U))
        jn = np.linalg.norm(J0, axis=1)
        jn = jn[jn > 0]
        if jn.size and g0 > 0:
            rho = float(np.clip(g0 / np.median(jn), rho0, rho_max))
    prev_worst = np.inf

    # Radius carried between rounds. The multipliers move, but the curvature
    # of the merit does not change discontinuously with them, so the length the
    # last round settled on is a far better guess than a fixed fraction of the
    # iterate. Reset each round instead, every round spends its first several
    # iterations shrinking by a factor of four back to the value the previous
    # one had already found:
    radius_carry = None

    # Shortest step seen, and the iterate that produced it:
    U_best = U.copy()
    best_step = np.inf

    # Best round overall, since a later round may wander away from a good answer:
    U_keep = None
    keep_key = None

    # Endpoint multiplier, exposed so callers can check it against a KKT solve:
    system._costate = None

    # Augmented-Lagrangian penalty at a control:
    def penalty(V, Z=None):
        h, _ = evaluate(V, Z=Z)
        if h.size == 0:
            return 0.0, -np.inf
        eta = np.maximum(0.0, mu + rho * h)
        return float(np.sum(eta ** 2 - mu ** 2)) / (2.0 * rho), float(h.max())

    # Merit at a control: cost, the augmented-Lagrangian penalty, and the
    # endpoint miss at a weight above the endpoint multiplier. One function for
    # both cases -- with no inequality rows the penalty term is identically
    # zero, so the constrained and unconstrained solves are the same test:
    def merit_at(V, sigma):
        Z_v = system.rollout(V)
        if not np.all(np.isfinite(Z_v)):
            return np.inf
        pen, _ = penalty(V, Z=Z_v)
        return cost_of(V) + pen \
            + sigma * np.linalg.norm(Z_v[-1][tidx] - zt)

    # Outer rounds, one multiplier update each:
    for outer_it in range(outer if con is not None else 1):
        held = None
        radius = (radius_carry if radius_carry is not None
                  else radius0 * max(tr_norm(U), 1.0))
        U_jac = U.copy()
        stale = False
        lam_prev = None

        # Inner rounds, minimizing the subproblem at the current multipliers:
        for inner_it in range(inner if con is not None else max_it):
            if not np.all(np.isfinite(U)):
                U = U_best
                break

            # Gradient of the merit's smooth part, where one is available. The
            # unconstrained dual path may be given a cost without its gradient,
            # and predicts its step from the model below instead:
            g = None
            if use_rec:
                e, delta, lam, dU = _recursive_step(system, U, zt, R_N_inv,
                                                    reg, m, tidx)
                if not np.all(np.isfinite(e)):
                    U = U_best
                    break
                step_norm = np.linalg.norm(dU)
                if not np.isfinite(step_norm):
                    U = U_best
                    break
                system._costate = np.array(lam, float)
                sigma = 2.0 * float(np.max(np.abs(lam))) + 1.0
                g = cost_grad_of(U)
            else:
                e, Co = system.endpoint_jac(U)
                if not np.all(np.isfinite(e)):
                    U = U_best
                    break
                delta = e - zt

                # General cost with no inequality rows: the exact dual solve,
                # which needs no forward gradient because U is dinv(Co' lam)
                # by construction at every full step:
                if cost is not None and con is None:
                    d_tgt = Co @ U - delta

                    # Seed from the quadratic solution, warm start thereafter:
                    if lam_prev is None:
                        lam_prev = safe_solve(Co @ Co.T + reg * np.eye(m),
                                              2.0 * d_tgt)

                        # Scale the seed up until the map responds, past any threshold:
                        for _ in range(80):
                            if np.linalg.norm(f_dinv(Co.T @ lam_prev)) > 1e-6:
                                break
                            lam_prev = lam_prev * 2.0

                    def _res(L):
                        return Co @ f_dinv(Co.T @ L) - d_tgt

                    def _jac(L):
                        s_ = Co.T @ L
                        if f_dprime is not None:
                            dp = np.asarray(f_dprime(s_))
                        else:
                            h_ = 1e-6 * (1.0 + np.abs(s_))
                            dp = (f_dinv(s_ + h_)
                                  - f_dinv(s_ - h_)) / (2.0 * h_)

                        # Flat dp is a diagonal M, (N,nu,nu) dp a block diagonal one:
                        if dp.ndim == 1:
                            return Co @ (dp[:, None] * Co.T)
                        CoB = Co.reshape(m, -1, system.nu)
                        return np.einsum("pki,kij,qkj->pq", CoB, dp, CoB)

                    # Walk out to the target, each step warm starting the next:
                    lam = lam_prev
                    for frac in (0.25, 0.5, 1.0):
                        sol = root(lambda L, fr=frac:
                                   Co @ f_dinv(Co.T @ L) - fr * d_tgt,
                                   lam, jac=_jac, method="hybr")
                        lam = sol.x
                    ok_dual = False
                    if np.all(np.isfinite(lam)):
                        ok_dual = (np.linalg.norm(_res(lam))
                                   <= 1e-6 * max(1.0, np.linalg.norm(d_tgt)))

                    if ok_dual:
                        lam_prev = lam
                        U_new = f_dinv(Co.T @ lam)
                    else:

                        # Dual solve missed, so take the quadratic step and retry later:
                        W_q = Co @ Co.T + reg * np.eye(m)
                        lam = safe_solve(W_q, 2.0 * d_tgt)
                        U_new = Co.T @ lam / 2.0

                    if not np.all(np.isfinite(U_new)):
                        U = U_best
                        break

                    system._costate = np.array(lam, float)
                    dU = U_new - U
                    step_norm = np.linalg.norm(dU)

                    # A deadband can return a zero step on the first pass, so
                    # push past the threshold rather than call it converged:
                    if (step_norm < 1e-14 * max(np.linalg.norm(U), 1.0)
                            and np.linalg.norm(delta) >= end_tol):
                        for _ in range(60):
                            lam_prev = lam_prev * 2.0
                            U_new = f_dinv(Co.T @ lam_prev)
                            if np.linalg.norm(U_new - U) > 1e-9:
                                break
                        dU = U_new - U
                        step_norm = np.linalg.norm(dU)
                        if step_norm < 1e-14:
                            U = U_best
                            break

                    sigma = 2.0 * float(np.max(np.abs(lam))) + 1.0
                    if f_grad is not None:
                        g = cost_grad_of(U)

                else:

                    # Cost gradient, with the active constraint rows folded in:
                    g = cost_grad_of(U)
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

                    # Metric at this iterate, read once and reused below:
                    apply_M = metric_at(U)

                    # Active rows deform the metric to (H + rho G'G)^-1:
                    if G.shape[0]:
                        n_act = G.shape[0]

                        # G M, formed once, since M is symmetric:
                        G_M = apply_M(G.T).T
                        M_act = np.eye(n_act) / rho + G_M @ G.T \
                            + 1e-12 * np.eye(n_act)

                        # Woodbury, so the solve is the size of the active set:
                        def apply_R_eff_inv(V, G=G, G_M=G_M, M_act=M_act):
                            corr = G.T @ safe_solve(M_act, G_M @ V)
                            return apply_M(V - corr)
                    else:

                        # No active rows, so the metric is the cost Hessian inverse itself:
                        def apply_R_eff_inv(V):
                            return apply_M(V)

                    # Weighted output controllability Gramian and the lambda step:
                    R_inv_Co = apply_R_eff_inv(Co.T)
                    W_R = Co @ R_inv_Co + reg * np.eye(m)
                    R_inv_grad = apply_R_eff_inv(g)

                    # Symmetric scaling, so conditioning stops depending on N:
                    d_W = 1.0 / np.sqrt(np.maximum(np.diag(W_R), 1e-300))
                    W_s = (W_R * d_W[:, None]) * d_W[None, :]
                    lam = safe_solve(W_s, (Co @ R_inv_grad - delta) * d_W) * d_W
                    dU = -(R_inv_grad - R_inv_Co @ lam)
                    step_norm = np.linalg.norm(dU)
                    if not np.isfinite(step_norm):
                        U = U_best
                        break

                    system._costate = np.array(lam, float)

                    # An exact penalty needs a weight above the endpoint multiplier:
                    sigma = 2.0 * float(np.max(np.abs(lam))) + 1.0

            # Track the shortest step, and stop once it is small enough:
            if step_norm < best_step:
                best_step = step_norm
                U_best = U.copy()
            # A short step means the subproblem is solved. The endpoint is
            # only checked alongside it on the dual path, where a deadband can
            # return a zero step at a missed endpoint; elsewhere the endpoint
            # is closed by the restore below, and demanding it here stops the
            # inner loop breaking early -- which is also how the outer loop
            # detects a stalled round, so it would run every round to its cap:
            if step_norm < ftol * max(np.linalg.norm(U), 1.0) \
                    and (con is not None or cost is None
                         or np.linalg.norm(delta) < end_tol):
                break

            # Merit at the current iterate:
            merit = merit_at(U, sigma)

            # Clip the step to the radius, in the metric the cost is measured in:
            p = dU
            p_norm = tr_norm(p)
            frac = 1.0
            if p_norm > radius:
                frac = radius / p_norm
                p = p * frac

            # One evaluation, rather than a search over step lengths. The
            # predicted reduction is linear in the step where a gradient is
            # available; the dual path, which may be given a cost without one,
            # reads it off the model the step was built from, whose endpoint
            # lands on the target in proportion to how much of it was taken:
            U_try = U + p
            merit_try = merit_at(U_try, sigma)
            if g is not None:
                predicted = -float(g @ p)
            else:
                predicted = merit - (cost_of(U_try)
                                     + sigma * (1.0 - frac)
                                     * np.linalg.norm(delta))

            moved = False
            if np.isfinite(merit_try):
                achieved = merit - merit_try

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

            # Optional trace of the radius schedule:
            if getattr(system, "_count_ls", False):
                print("[shoot] it %4d  radius %9.3e  step %9.3e  "
                      "merit %12.6g  %s"
                      % (inner_it, radius, step_norm, merit,
                         "ok" if moved else "rej"))

            # A rejected step just means try shorter, or rebuild the Jacobian.
            # Below a radius this small against the iterate the model permits
            # no step worth taking, and shrinking further only repeats the same
            # rejection: the multiplier update waiting outside is worth more
            # than another dozen halvings down to an absolute floor:
            if not moved:
                if radius > 1e-6 * max(tr_norm(U), 1.0):
                    continue
                if not stale and con is not None:
                    stale = True
                    radius = radius0 * max(tr_norm(U), 1.0)
                    continue
                break

        # Hand the calibrated radius to the next round:
        radius_carry = radius

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
               cost_of(U))
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

    # Constrained, closing the endpoint alone pushes back through the active rows:
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

    # Polish: slide along the constraint surface to reduce cost:
    if con is not None:
        for _ in range(polish):
            h, Jh = evaluate(U, want_jac=True)
            _, Co = system.endpoint_jac(U)
            act = np.where(h > -tol_act)[0] if h.size else np.array([], int)
            A = np.vstack([Co, Jh[act]]) if act.size else Co

            # Cost gradient with endpoint and active-row motion projected out,
            # in the same metric the step above was taken in:
            apply_M = metric_at(U)
            g = cost_grad_of(U)
            R_inv_g = apply_M(g)
            R_inv_A = apply_M(A.T)
            gram = A @ R_inv_A + reg * (np.trace(A @ R_inv_A) / A.shape[0] + 1.0) \
                * np.eye(A.shape[0])
            p = -(R_inv_g - R_inv_A @ safe_solve(gram, A @ R_inv_g))
            p_norm = tr_norm(p)
            if p_norm < 1e-10 * max(tr_norm(U), 1.0):
                break

            # Where the solve stands, so a step cannot give any of it up:
            cost_now = cost_of(U)
            end_now = np.linalg.norm(system.endpoint(U) - zt)
            worst_now = float(h.max()) if h.size else -np.inf
            moved = False

            # The useful move is a short slide, so the ladder reaches far smaller steps:
            for a in [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]:
                U_try = U + a * p
                Z_try = system.rollout(U_try)
                if not np.all(np.isfinite(Z_try)):
                    continue
                h_try, _ = evaluate(U_try)
                if float(h_try.max()) > max(worst_now, tol_feas):
                    continue

                # First-order projection drifts, so allow it and restore below:
                if np.linalg.norm(Z_try[-1][tidx] - zt) > max(end_now * 10.0, end_tol):
                    continue
                if cost_of(U_try) < cost_now:
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