# Import packages:
import numpy as np
import casadi as ca
from .newton_shoot import newton_shoot
from .bounds import expand_weights, weighted_cost, weighted_cost_grad, safe_solve


# Compile a list of constraint expressions into evaluators:
def compile_constraints(system, funcs):

    # A constraint is any expression g(z, u) <= 0, applied at every node.  What
    # it means is irrelevant here: a keep-out circle, an attitude limit and a
    # thrust bound are the same object and are handled identically.
    #
    # The gradient is never taken through the rollout.  Each constraint reports
    # which states and controls it actually touches, so only its local gradient
    # is differentiated and then chained with dz/dU, which the system already
    # compiles and caches.  Differentiating through the horizon instead costs
    # about a hundred times more to build, which matters when the constraint set
    # changes between solves:
    if not funcs:
        return None
    z = ca.MX.sym("z", system.nx)
    u = ca.MX.sym("u", system.nu)
    K = system.N + 1
    blocks = []
    for f in funcs:
        expr = f(z, u)
        s_dep = [i for i, b in enumerate(ca.which_depends(expr, z, 1, False)) if b]
        u_dep = [i for i, b in enumerate(ca.which_depends(expr, u, 1, False)) if b]
        Jz = ca.jacobian(expr, z)
        Ju = ca.jacobian(expr, u)

        # Values and gradients are separate maps.  The line search asks for
        # values an order of magnitude more often than the step asks for
        # gradients, and one function with three outputs would compute both
        # local Jacobians every time a value was wanted:
        g_val = ca.Function("g", [z, u], [expr])
        g_jac = ca.Function("gj", [z, u],
                            [Jz[:, s_dep] if s_dep else ca.MX.zeros(1, 0),
                             Ju[:, u_dep] if u_dep else ca.MX.zeros(1, 0)])
        f_val = ca.Function.map(g_val, K)
        f_jac = ca.Function.map(g_jac, K)
        blocks.append((s_dep, u_dep, f_val, f_jac))

    # Union of the states any constraint needs, so one row Jacobian serves all:
    states = tuple(sorted({i for b in blocks for i in b[0]}))
    return blocks, states, K


# Evaluate every constraint at a control sequence:
def eval_constraints(system, con, U, want_jac=False):

    # Values come from the cached rollout, so no dynamics are re-integrated:
    blocks, states, K = con
    N, nu = system.N, system.nu
    Z = np.asarray(system.rollout(U), float)

    # The terminal node has no control of its own, so it reuses the last one:
    idx = [min(k, N - 1) for k in range(K)]
    Un = np.stack([U[i * nu:(i + 1) * nu] for i in idx], axis=1)
    JZ = system.row_jac(U, states) if (want_jac and states) else None


    h_all = []
    J_all = []
    for s_dep, u_dep, f_val, f_jac in blocks:
        h_all.append(np.array(f_val(Z.T, Un)).flatten())
        if not want_jac:
            continue
        gz, gu = f_jac(Z.T, Un)

        # Chain the local state gradient with dz/dU:
        rows = np.zeros((K, N * nu))
        if s_dep:
            cols = [states.index(i) for i in s_dep]
            gz = np.array(gz).reshape(K, len(s_dep))
            rows += np.einsum("kj,kjd->kd", gz, JZ[:, cols, :])

        # A control only affects its own node, so its gradient is placed
        # directly rather than chained through anything:
        if u_dep:
            gu = np.array(gu).reshape(K, len(u_dep))
            for k in range(K):
                base = idx[k] * nu
                for c, j in enumerate(u_dep):
                    rows[k, base + j] += gu[k, c]
        J_all.append(rows)


    h = np.concatenate(h_all) if h_all else np.zeros(0)
    return h, (np.vstack(J_all) if (want_jac and J_all) else None)


# Nudge a starting trajectory off a symmetric saddle:
def _break_symmetry(system, con, U, n_dof):

    # A violated constraint whose gradient vanishes leaves the solve with no
    # direction to move in.  A keep-out entered dead centre is the usual case:
    # the expression is radially symmetric, so its gradient is exactly zero
    # there and the trajectory feels no push either way.
    #
    # The fix belongs in the starting point, not in the derivative.  Substituting
    # a direction into the constraint Jacobian would corrupt the multiplier
    # update, the active set and the metric, all of which take those rows to be
    # dh/dU.  Nudging the trajectory instead leaves every one of them honest: one
    # step off the saddle and the real gradient is nonzero from then on.
    blocks, states, K = con
    h, J = eval_constraints(system, con, U, want_jac=True)
    if h.size == 0:
        return U
    viol = h > 0.0
    if not viol.any():
        return U

    # Rows that are violated but carry almost no gradient, measured against the
    # violated rows that do:
    rn = np.linalg.norm(J, axis=1)
    ref = np.median(rn[viol])
    if not np.any(viol & (rn < 0.05 * max(ref, 1e-12))):
        return U

    # Escape sideways.  Pushing along the path does not leave a region centred
    # on it, and on a vehicle whose downrange authority dwarfs its lateral
    # authority the most controllable state is exactly the one that does not
    # help, so the travel direction is removed first and the most controllable
    # of what remains is taken:
    Z = np.asarray(system.rollout(U))
    JZ = system.row_jac(U, states) if states else None
    if JZ is None:
        return U
    k = int(np.argmax(np.where(viol, h, -np.inf)) % K)
    k_prev = max(k - 1, 0)
    reach = np.linalg.norm(JZ[k], axis=1)
    if reach.max() <= 0.0:
        return U
    travel = Z[k, list(states)] - Z[k_prev, list(states)]
    t_norm = np.linalg.norm(travel)
    direction = reach / reach.max()
    if t_norm > 1e-12:
        unit = travel / t_norm
        direction = direction - float(direction @ unit) * unit

    # Perfectly symmetric leaves nothing to prefer, so take a perpendicular and
    # a fixed sign, which keeps repeated solves of the same problem in agreement:
    if np.linalg.norm(direction) < 1e-9:
        if t_norm > 1e-12 and len(states) >= 2:
            direction = np.zeros(len(states))
            direction[0], direction[1] = -travel[1], travel[0]
        else:
            direction = np.zeros(len(states))
            direction[int(np.argmax(reach))] = 1.0
    n = np.linalg.norm(direction)
    if n < 1e-12:
        return U
    direction = direction / n
    if direction[int(np.argmax(np.abs(direction)))] < 0.0:
        direction = -direction

    # Least-norm control change that moves that node the requested way, sized to
    # the violation so a deep overlap gets a proportionally larger nudge:
    Jk = np.einsum("j,jd->d", direction, JZ[k])
    jn = float(Jk @ Jk)
    if jn < 1e-18:
        return U
    step = float(np.sqrt(max(h[viol].max(), 0.0))) + 1e-3
    return U + Jk * (step / jn)


# Minimum-effort shoot to a target, subject to inequality constraints:
def lambda_shoot(system, z_target, constraints=(), U0=None, R_weights=None,
                 max_it=1200, ftol=1e-6, outer=25, inner=10):

    # Settings that are not the caller's business.  Every one of these was swept
    # across twelve systems and the solve is insensitive to all of them, so they
    # are constants rather than arguments: exposing a knob nobody needs to turn
    # only invites someone to turn it.
    reg = 1e-10                 # regularization on the Gramian solve
    depth = 8                   # Anderson memory, unconstrained path only
    rho0, rho_max = 10.0, 1e8   # augmented-Lagrangian penalty and its cap
    tol_feas = 1e-6             # a violation this small counts as satisfied
    tol_act = 1e-4              # a row this close to its limit counts as active
    tol_c = 1e-3                # complementarity
    jac_tol = 0.05              # drift, relative to |U|, that stales a Jacobian
    radius0, radius_max = 0.1, 1e6      # trust radius, relative to |U|
    accept_ratio, shrink, grow = 0.1, 0.25, 2.0

    # Only focus on constrained states (i.e., ones we are focusing on for endpoint):
    zt = system.target(z_target)
    m = system.m
    tidx = list(system.tidx)
    n_dof = system.N * system.nu

    # Stacked control weight R_N, and its inverse:
    r_diag = expand_weights(R_weights, system.N, system.nu)
    R_N = 2.0 * r_diag
    R_N_inv = 1.0 / R_N

    # Compile whatever constraints were given into one block, cached on the
    # system so a repeated solve with the same constraints does not rebuild the
    # graph.  Building it is comparable to the solve itself:
    funcs = list(constraints)
    if funcs:
        # Keyed by the function objects themselves, not their ids: an id is
        # reused once a lambda is collected, which would hand back a graph built
        # for entirely different constraints:
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

    # Newton shoot for a feasible control sequence.  It knows nothing about the
    # constraints, so it can land outside them, and on a system with a
    # singularity just past a bound -- a steering angle beyond ninety degrees,
    # say -- the augmented Lagrangian can never pull it back, because doing so
    # means passing through the singularity.  The starting point is therefore
    # kept inside any bound the caller declared:
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


    # Multipliers, one per constraint row, and the penalty weight:
    h0, _ = evaluate(U)
    mu = np.zeros(h0.size)

    # Scale the penalty to the problem rather than starting every solve at the
    # same number.  rho multiplies a constraint gradient and is compared against
    # the cost gradient, so the ratio of their magnitudes is the scale at which
    # the two terms are comparable.  Starting far below it wastes outer rounds
    # climbing, which on a large system is most of the solve time:
    rho = rho0
    if h0.size:
        _, J0 = evaluate(U, want_jac=True)
        g0 = np.linalg.norm(weighted_cost_grad(U, r_diag))
        jn = np.linalg.norm(J0, axis=1)
        jn = jn[jn > 0]
        if jn.size and g0 > 0:
            rho = float(np.clip(g0 / np.median(jn), rho0, rho_max))
    prev_worst = np.inf

    # Weight on the endpoint term in the merit.  An exact penalty needs it to
    # exceed the endpoint multiplier, so it is read off the multiplier the step
    # already computes.  A fixed weight is silently wrong once rho passes it, at
    # which point every trial step is rejected and the violation stops falling:
    sigma = 1.0
    U_best = U.copy()
    best_step = np.inf

    # Augmented-Lagrangian penalty at a control:
    def penalty(V):
        h, _ = evaluate(V)
        if h.size == 0:
            return 0.0, -np.inf
        eta = np.maximum(0.0, mu + rho * h)
        return float(np.sum(eta ** 2 - mu ** 2)) / (2.0 * rho), float(h.max())

    # Outer rounds.  With no constraints there is nothing to update, so a single
    # pass is the whole solve:
    for outer_it in range(outer if con is not None else 1):
        hist_U, hist_F, held = [], [], None
        radius = radius0 * max(np.linalg.norm(U), 1.0)
        U_jac = U.copy()
        stale = False
        blend = 1.0
        prev_step = np.inf

        for inner_it in range(inner if con is not None else max_it):
            if not np.all(np.isfinite(U)):
                U = U_best
                break
            e, Co = system.endpoint_jac(U)
            if not np.all(np.isfinite(e)):
                U = U_best
                break
            delta = e - zt

            # Cost gradient, with the active rows folded in.  A row is active
            # when its shifted multiplier is positive, which is a smooth test
            # and so does not chatter between iterations.  The Jacobian is the
            # dearest call and moves slowly over a damped step, so it is held on
            # a schedule while values are refreshed every iteration:
            g = weighted_cost_grad(U, r_diag)
            G = np.zeros((0, n_dof))
            if con is not None:
                h, _ = evaluate(U)
                eta = np.maximum(0.0, mu + rho * h)

                # A row sitting on its limit belongs in the active set even when
                # its multiplier has decayed to zero.  Testing the multiplier
                # alone drops it, so nothing holds the trajectory there, the
                # metric loses the direction it must not move in, and the solve
                # oscillates on and off the constraint instead of settling on
                # it.  The value itself is the reliable test:
                act = np.where((eta > 0.0) | (h > -tol_act))[0]

                # Only the active rows enter the step, sliced from the held
                # Jacobian.  Compiling a function per subset was tried and costs
                # far more to build than the extra rows cost to evaluate:
                if act.size:

                    # Refresh the Jacobian on evidence rather than on a count.
                    # It stays valid while the trajectory has not moved far from
                    # where it was built, so the trigger is how far U has
                    # drifted since then, measured against its own size.  A
                    # rejected step is the other signal: the direction came from
                    # this Jacobian, so a step the merit will not take means the
                    # linearization no longer describes the problem.  A tight,
                    # curved landscape then refreshes often and a gentle one
                    # rarely, without either being chosen in advance:
                    drift = np.linalg.norm(U - U_jac)
                    if held is None or stale or \
                            drift > jac_tol * max(np.linalg.norm(U), 1.0):
                        _, held = evaluate(U, want_jac=True)
                        U_jac = U.copy()
                        stale = False
                    g = g + eta[act] @ held[act]
                    G = held[act]

            # Active rows deform the control weight to R_eff = R_N + rho G'G.
            # No second derivative is taken: G'G is an outer product of first
            # derivatives, the same construction as the Gramian below.  Woodbury
            # reduces the inverse to a solve the size of the active set:
            if G.shape[0]:
                n_act = G.shape[0]
                M_act = np.eye(n_act) / rho + (G * R_N_inv) @ G.T \
                    + 1e-12 * np.eye(n_act)

                def apply_R_eff_inv(V, G=G, M_act=M_act):
                    corr = G.T @ safe_solve(M_act, (G * R_N_inv) @ V)
                    return (R_N_inv[:, None] * (V - corr)) if V.ndim > 1 \
                        else R_N_inv * (V - corr)
            else:

                def apply_R_eff_inv(V):
                    return (R_N_inv[:, None] * V) if V.ndim > 1 else R_N_inv * V

            # Weighted output controllability Gramian and the lambda step.  With
            # no active rows R_eff is R_N and this is the least-norm control, in
            # the R_N metric, closing the observation gap:
            R_inv_Co = apply_R_eff_inv(Co.T)
            W_R = Co @ R_inv_Co + reg * np.eye(m)
            R_inv_grad = apply_R_eff_inv(g)
            lam = safe_solve(W_R, Co @ R_inv_grad - delta)
            dU = -(R_inv_grad - R_inv_Co @ lam)
            step_norm = np.linalg.norm(dU)
            if not np.isfinite(step_norm):
                U = U_best
                break

            # Endpoint weight for the merit.  An exact penalty needs it above
            # the endpoint multiplier, and it is recomputed each step rather
            # than ratcheted: the multiplier inflates with rho, so a weight that
            # only ever rises ends up dominating the merit entirely and no step
            # that reduces a constraint violation can pay for itself:
            sigma = 2.0 * float(np.max(np.abs(lam))) + 1.0

            if step_norm < best_step:
                best_step = step_norm
                U_best = U.copy()
            if step_norm < ftol * max(np.linalg.norm(U), 1.0):
                break

            # With no constraints the step is a fixed-point map, and the plain
            # iteration barely contracts on a long horizon, so the last few
            # residuals are mixed to cancel the slow mode.  With active rows the
            # step is already Newton on the augmented Lagrangian and mixing it
            # is unstable, so that path takes a line search instead:
            if con is None:
                if step_norm > prev_step:
                    hist_U.clear()
                    hist_F.clear()
                    blend = max(blend * 0.5, 1e-3)
                else:
                    blend = min(blend * 1.2, 1.0)
                prev_step = step_norm
                hist_U.append(U.copy())
                hist_F.append(dU.copy())
                if len(hist_U) > depth:
                    hist_U.pop(0)
                    hist_F.pop(0)
                k = len(hist_U)
                if k == 1:
                    U_new = U + blend * dU
                else:
                    dF = np.column_stack(
                        [hist_F[i + 1] - hist_F[i] for i in range(k - 1)])
                    dU_hist = np.column_stack(
                        [hist_U[i + 1] - hist_U[i] for i in range(k - 1)])
                    try:
                        gamma = np.linalg.lstsq(dF, dU, rcond=1e-10)[0]
                        U_new = U + blend * dU - (dU_hist + blend * dF) @ gamma
                    except np.linalg.LinAlgError:
                        U_new = U + blend * dU
                if not np.all(np.isfinite(U_new)) or \
                        np.linalg.norm(U_new - U) > 50.0 * max(step_norm, 1e-12):
                    hist_U.clear()
                    hist_F.clear()
                    blend = max(blend * 0.5, 1e-3)
                    U_new = U + blend * dU
                U = U_new
                continue

            # Trust region rather than a line search.  A backtracking search
            # spends one rollout per trial and can burn half a dozen before it
            # accepts anything.  A trust region evaluates once and uses the
            # result to size the next step, so the same information costs a
            # fraction of the evaluations:
            pen_now, _ = penalty(U)
            merit = weighted_cost(U, r_diag) + pen_now \
                + sigma * np.linalg.norm(delta)

            # Clip the step to the radius the model is currently trusted over:
            p = dU
            p_norm = np.linalg.norm(p)
            if p_norm > radius:
                p = p * (radius / p_norm)
            predicted = -float(g @ p)

            U_try = U + p
            Z_try = system.rollout(U_try)
            moved = False
            if np.all(np.isfinite(Z_try)):
                pen_try, _ = penalty(U_try)
                achieved = merit - (weighted_cost(U_try, r_diag) + pen_try
                                    + sigma * np.linalg.norm(Z_try[-1][tidx] - zt))

                # How much of the promised improvement the step delivered.  Near
                # one means the model holds over this distance and the radius can
                # grow; a loss means it does not and the radius must shrink:
                ratio = achieved / max(abs(predicted), 1e-16)

                # Accept only when the step delivered a decent share of what was
                # promised.  Taking any improvement at all lets the solve creep
                # along on steps the model does not really support, which ends
                # somewhere feasible but short of the optimum:
                if ratio > accept_ratio and achieved > 0.0:
                    U = U_try
                    moved = True
                    if ratio > 0.75 and p_norm >= radius:
                        radius = min(radius * grow, radius_max)
                elif ratio < 0.25:
                    radius = max(radius * shrink, 1e-10)
            else:
                radius = max(radius * 0.25, 1e-10)
            # A rejected step is not a failure: the radius has just been cut,
            # so the next pass tries a shorter one.  Only a radius that has
            # collapsed means there is nothing left to try, and a rebuilt
            # Jacobian is worth one attempt before concluding that:
            if not moved:
                if radius > 1e-9:
                    continue
                if not stale:
                    stale = True
                    radius = radius0 * max(np.linalg.norm(U), 1.0)
                    continue
                break

        if con is None:
            U = U_best
            break

        # Multiplier update, and the complementarity measure.  A multiplier left
        # active on a row standing clear of its limit is what freezes an
        # unnecessary margin, so convergence requires it to have decayed:
        h, _ = evaluate(U)
        worst = float(h.max()) if h.size else -np.inf
        mu = np.maximum(0.0, mu + rho * h)
        slack = float(np.max(mu * np.maximum(0.0, -h))) if h.size else 0.0

        # One multiplier update has to happen before the loop may exit, or the
        # penalty alone shaped the answer and complementarity was never tested:
        if outer_it > 0 and worst < tol_feas and slack < tol_c:
            break

        # Complementarity alone is not enough to stop on.  The slack measure is
        # mu times the remaining margin, and mu grows with rho, so on a problem
        # that settles exactly on a constraint it never falls below the
        # tolerance.  Once the violation is converged and the inner solve has
        # stopped moving, there is nothing left to gain and raising rho only
        # destroys the conditioning:
        if outer_it > 0 and worst < tol_feas and inner_it <= 1:
            break

        # Tighten only when the worst violation did not fall enough.  The cap
        # matters: a violation that stalls at large rho is an inner-loop failure,
        # and raising rho further only destroys the conditioning:
        if worst > 0.5 * prev_worst:
            rho = min(rho * 4.0, rho_max)
        prev_worst = worst

    # Final restoration.  A constraint-blind Newton shoot cannot be used here:
    # at convergence the trajectory sits on the active constraints, so anything
    # that moves it purely to close the endpoint pushes straight back through
    # them.  The endpoint residual and the active violations are therefore
    # driven down together, by the same least-norm projection the step uses --
    # the active rows are simply stacked onto the endpoint Jacobian, so the
    # Gramian being inverted is C_o R^-1 C_o' with those rows included:
    if con is None:
        U = newton_shoot(system, zt, U)
    else:
        for _ in range(25):
            e, Co = system.endpoint_jac(U)
            r = e - zt
            h, Jh = evaluate(U, want_jac=True)

            # Rows that are active or close to it, so a constraint about to be
            # entered is held rather than crossed:
            act = np.where(h > -tol_feas)[0] if h.size else np.array([], int)
            resid = np.concatenate([r, np.maximum(h[act], 0.0)]) if act.size else r
            if np.linalg.norm(resid) < 1e-11:
                break
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
            # A rejected step is not a failure: the radius has just been cut,
            # so the next pass tries a shorter one.  Only a radius that has
            # collapsed means there is nothing left to try, and a rebuilt
            # Jacobian is worth one attempt before concluding that:
            if not moved:
                if radius > 1e-9:
                    continue
                if not stale:
                    stale = True
                    radius = radius0 * max(np.linalg.norm(U), 1.0)
                    continue
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