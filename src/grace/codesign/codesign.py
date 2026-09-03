# Import packages and shooting:
import os
import hashlib
import numpy as np
from ..shooting.bounds import safe_solve
import casadi as ca
from scipy.optimize import brentq, minimize
from ..shooting.lambda_shoot import lambda_shoot

# Value of the running cost at a control:
def _cost_val(cost, U):

    # No cost given means the quadratic minimum-effort default:
    if cost is None:
        return float(np.asarray(U) @ np.asarray(U))
    return float(cost[0](U))

# Cost gradient at a control, when the cost supplies one:
def _cost_grad(cost, U):

    # The quadratic default differentiates to twice the control:
    if cost is None:
        return 2.0 * np.asarray(U, float).flatten()

    # A general cost only carries its gradient if the caller passed one:
    if len(cost) > 3 and cost[3] is not None:
        return np.asarray(cost[3](U), float).flatten()
    return None

# Endpoint multiplier at an inner optimum:
def _costate_at(sysp, U, Co, cost):

    # Stationarity of the inner problem is grad_U J = Co' lam, so when the cost
    # gradient is available the multiplier is recovered by a least-squares solve
    # at the returned control -- exactly what the quadratic path did with 2U,
    # and correct for any cost:
    g = _cost_grad(cost, U)
    if g is not None:
        m = Co.shape[0]
        return safe_solve(Co @ Co.T + 1e-12 * np.eye(m), Co @ g)

    # Without a gradient, fall back on the multiplier the inner solve produced.
    # It uses the same convention, but it is the value at the last inner step
    # rather than at the polished control:
    lam = getattr(sysp, "_costate", None)
    if lam is None:
        raise ValueError(
            "a general cost needs either its gradient, passed as the fourth "
            "element of the cost tuple, or an inner solve that stored a "
            "costate; neither was available")
    return np.asarray(lam, float).flatten()

# Build a parameterized system family and its parameter-sensitivity rollout:
def _build_param_family(dynamics, nx, nu, N, z0, dt, param_name, substeps=1, jit=True,
                        target_idx=None, jit_flags="-O1", cache_dir=".grace_cache",
                        n_param=1):

    # Symbolic state, control, and the design parameters. n_param is one for a
    # scalar design and more when several are sized together; nothing else in
    # the construction changes, since the parameter only ever enters through
    # the dynamics.
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    p = ca.MX.sym(param_name, n_param)

    # Integrate one control period with RK4, carrying the parameter:
    h = dt / substeps
    zc = z
    for _ in range(substeps):
        k1 = dynamics(zc, u, p)
        k2 = dynamics(zc + 0.5 * h * k1, u, p)
        k3 = dynamics(zc + 0.5 * h * k2, u, p)
        k4 = dynamics(zc + h * k3, u, p)
        zc = zc + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    step = ca.Function("step", [z, u, p], [zc])

    # Build the full rollout with a compiled scan:
    U = ca.MX.sym("U", N * nu)
    Useq = ca.reshape(U, nu, N)
    acc = step.mapaccum("roll", N)
    Zc = acc(ca.DM(z0), Useq, ca.repmat(p, 1, N))
    Zmat = ca.horzcat(ca.DM(z0), Zc).T

    # Constrain only the requested endpoint components:
    tidx = list(range(nx)) if target_idx is None else list(target_idx)
    gend = Zmat[-1, tidx].T

    # Symbolic functions: F_end and its control Jacobian are always evaluated
    # together, so they share one function -- one rollout instead of two:
    F_ej = ca.Function("F_ej_p", [U, p], [gend, ca.jacobian(gend, U)])
    F_end = ca.Function("F_end_p", [U, p], [gend])
    F_dp = ca.Function("F_dp_p", [U, p], [ca.jacobian(gend, p)])
    F_roll = ca.Function("F_roll_p", [U, p], [Zmat])

    # === COMPILED FAMILY CACHE ===
    # The expression graph fully determines the generated code, so hash it: an
    # unchanged problem reuses the shared library instead of recompiling.
    if jit:
        sig = f"{str(gend)}|{str(Zmat)}|{jit_flags}|{n_param}"
        key = hashlib.sha1(sig.encode()).hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        stem = f"fam_{key}"
        c_path = os.path.join(cache_dir, stem + ".c")
        so_path = os.path.join(cache_dir, stem + ".so")

        # Generate and compile only when this problem has not been seen:
        if not os.path.exists(so_path):
            gen = ca.CodeGenerator(stem + ".c")
            for F in (F_ej, F_end, F_dp, F_roll):
                gen.add(F)
            gen.generate(cache_dir + os.sep)
            cc = os.environ.get("CC", "gcc")
            rc = os.system(f"{cc} {jit_flags} -shared -fPIC "
                           f"-o {so_path} {c_path} 2>/dev/null")

            # A failed compile is not fatal -- fall back to the MX functions:
            if rc != 0 or not os.path.exists(so_path):
                print("[codesign] WARNING: family compile failed, "
                      "running interpreted")
                so_path = None

        # Load the compiled entry points:
        if so_path is not None:
            F_ej = ca.external("F_ej_p", so_path)
            F_end = ca.external("F_end_p", so_path)
            F_dp = ca.external("F_dp_p", so_path)
            F_roll = ca.external("F_roll_p", so_path)

    # Return the compiled parameter family (with the targeted endpoint indices):
    return dict(step=step, F_ej=F_ej, F_end=F_end, F_dp=F_dp, F_roll=F_roll,
                tidx=tidx)

# A thin System-like wrapper that pins the family at one parameter value:
class _PinnedSystem:

    # Pin the parameter family at a value so shooting solvers can run on it:
    def __init__(self, fam, nx, nu, N, z0, dt, pval):
        self.fam = fam
        self.nx = nx
        self.nu = nu
        self.N = N
        self.z0 = np.asarray(z0, float)
        self.dt = dt
        self.tidx = fam.get("tidx", list(range(nx)))
        self.m = len(self.tidx)
        self.pos_jac = None
        self.pval = np.atleast_1d(np.asarray(pval, float))
        self._count_iters = False
        self._plain_step = False

    # Roll out at the pinned parameter:
    def rollout(self, U):
        return np.array(self.fam["F_roll"](np.asarray(U).flatten(), self.pval))

    # Endpoint and control Jacobian at the pinned parameter:
    def endpoint_jac(self, U):
        g, J = self.fam["F_ej"](np.asarray(U).flatten(), self.pval)
        return np.array(g).flatten(), np.array(J)

    def endpoint(self, U):
        return np.array(self.fam["F_end"](np.asarray(U).flatten(), self.pval)).flatten()

    def target(self, z_target):
        zt = np.asarray(z_target, float)
        return zt[self.tidx] if len(zt) == self.nx else zt

# Trace the front by solving directly at each design value:
def scan(dynamics, nx, nu, N, z0, dt, target, param_name, objective, p_values,
         substeps=1, save="figures", job="scan", plot=True, target_idx=None,
         filter_dominated=True, cost=None):

    # Build the parameter family once:
    fam = _build_param_family(dynamics, nx, nu, N, z0, dt, param_name, substeps,
                              target_idx=target_idx)

    # Reduce the target to the constrained components:
    zt = np.asarray(target, float)
    if zt.size == nx:
        zt = zt[fam["tidx"]]

    # Solve at each design, warm-starting from the previous one:
    front = []
    U = None
    for pv in np.asarray(p_values, float):
        sp = _PinnedSystem(fam, nx, nu, N, z0, dt, float(pv))
        U = lambda_shoot(sp, zt, U0=U, cost=cost)
        front.append(dict(param=float(pv), cost=_cost_val(cost, U),
                          objective=float(objective(float(pv))),
                          control=np.array(U, copy=True)))

    # Drop dominated designs:
    if filter_dominated:
        front = pareto_front(front)
        front.sort(key=lambda f: f["objective"])

    # Plot the front:
    if plot:
        _plot_front(front, save, job)

    # Return the front:
    return front

# Run codesign over a named design parameter:
def _codesign_vector(fam, tidx, nx, nu, N, z0, dt, target, objective,
                     p0, p_bounds, weights, norm, rho, p_tol, max_outer,
                     save, job, plot, n_anchor, debug_vec=True, cost=None,
                     cost_dp=None):

    # Several design parameters at once. The inner solve is unchanged -- an
    # exact minimum-effort shoot at each candidate design -- and the design
    # gradient comes from the costate it already produces:
    #
    #     dJ*/dp = -lam' (de/dp) + dJ/dp
    #
    # At the inner optimum the control is stationary, so a design perturbation
    # only moves the endpoint constraint and the control's own contribution
    # drops out. The second term survives only when the running cost depends on
    # the design explicitly, which is what cost_dp supplies; for a cost written
    # in the control alone it is zero and this reduces to the contraction the
    # quadratic path always used. Nothing is differentiated through the solve.
    lo = np.array([b[0] for b in p_bounds], float)
    hi = np.array([b[1] for b in p_bounds], float)
    zt = np.asarray(target, float)[tidx]
    weights = np.linspace(0.0, 1.0, 11) if weights is None else np.asarray(weights)

    # Warm starts kept per design and chosen by nearest neighbour. The outer
    # solve does not walk smoothly through the design space -- a line search
    # can jump and come back -- so the last design evaluated is often not a
    # near neighbour of the next, and a control history optimal for a different
    # vehicle costs more to correct than it saves.
    warm = {"pts": []}
    stats = {"solves": 0}
    span = np.maximum(hi - lo, 1e-12)

    def _warm_for(p):
        if not warm["pts"]:
            return None
        P = np.array([q for q, _ in warm["pts"]])
        k = int(np.argmin(np.linalg.norm((P - p) / span, axis=1)))
        return warm["pts"][k][1] if \
            np.linalg.norm((P[k] - p) / span) < 0.15 else None

    def inner(p, ftol=1e-7, max_it=400):
        pc = np.clip(p, lo, hi)
        sysp = _PinnedSystem(fam, nx, nu, N, z0, dt, pc)

        # Tolerance matters more than it looks during the search. The outer
        # solve is an SQP reading the inner cost through a constraint; a
        # loosely converged inner solve makes that constraint noisy, and an SQP
        # responds to noise by deciding it has converged.
        #
        # The anchor sweep is different. It only places the ideal and nadir
        # points the weights are normalized against, which needs two digits and
        # not seven, so it passes a loose tolerance and is much cheaper.
        U = np.asarray(lambda_shoot(sysp, zt, U0=_warm_for(pc),
                                    ftol=ftol, max_it=max_it,
                                    cost=cost)).flatten()
        stats["solves"] += 1
        warm["pts"].append((pc.copy(), U.copy()))
        return sysp, U

    def cost_and_grad(p, **kw):
        pc = np.clip(p, lo, hi)
        sysp, U = inner(pc, **kw)
        _, Co = sysp.endpoint_jac(U)
        m = Co.shape[0]

        # The multiplier is read from the cost's own stationarity, so the
        # gradient below carries no assumption that the cost is quadratic:
        lam = _costate_at(sysp, U, Co, cost)
        dgp = np.array(fam["F_dp"](U, pc)).reshape(m, -1)
        gt = -(dgp.T @ lam)

        # Explicit design dependence of the running cost, when there is one:
        if cost_dp is not None:
            gt = gt + np.asarray(cost_dp(U, pc), float).flatten()
        return _cost_val(cost, U), gt, U

    # Anchor sweep, to place the ideal and nadir points the weights are
    # normalized against. The scalar path can sweep a dense grid because the
    # design is one number; in several dimensions a grid is out of reach and a
    # handful of corners is not enough -- a misplaced nadir makes one objective
    # negligible at every weight, and the whole front collapses onto a point.
    #
    # A scrambled low discrepancy sample covers the box far more evenly than
    # random points at this count, and the corners and the starting design are
    # included because the extremes of both objectives usually sit there.
    # Sobol is balanced only at powers of two, so the count is rounded to one:
    # Enough to place the ideal and nadir in a few dimensions without turning
    # the sweep into the dominant cost. Rounded to a power of two, which is
    # where Sobol is balanced.
    n_s = int(2 ** np.ceil(np.log2(max(8 * len(lo), 16))))
    try:
        from scipy.stats import qmc
        pts = qmc.Sobol(d=len(lo), scramble=True, seed=0).random(n_s)
    except Exception:
        pts = np.random.default_rng(0).random((n_s, len(lo)))
    anchors = [p0, lo.copy(), hi.copy()]
    anchors += [lo + q * (hi - lo) for q in pts]
    for i in range(len(lo)):
        a = p0.copy()
        a[i] = lo[i]
        anchors.append(a)
        b = p0.copy()
        b[i] = hi[i]
        anchors.append(b)
    Cg = np.array([cost_and_grad(a, ftol=1e-3, max_it=120)[0]
                   for a in anchors])
    Dg = np.array([float(objective(a)) for a in anchors])

    # Ideal and nadir, the same construction the scalar path uses. The nadir is
    # the cost at the design that minimizes the other objective, not the worst
    # cost anywhere in the sweep: a single poor anchor would otherwise set a
    # range so wide that the normalized cost is negligible at every sensible
    # design, the other term dominates at every weight, and the front collapses
    # to a point.
    C_id = Cg.min() - 0.01 * np.ptp(Cg)
    D_id = Dg.min() - 0.01 * np.ptp(Dg)
    C_rng = max(Cg[int(np.argmin(Dg))] - C_id, 1e-12)
    D_rng = max(Dg[int(np.argmin(Cg))] - D_id, 1e-12)

    h = 1e-6

    # Memoized, because an SQP asks for the objective, its gradient, the
    # constraints and their Jacobian as four separate calls at the same point,
    # and each would otherwise repeat the inner solve underneath. Caching turns
    # four solves per iteration into one, and the line search reuses points as
    # well.
    _pcache = {}

    def parts(p):
        # Rounded finely enough to catch a repeated evaluation and no finer.
        # A coarse key makes nearby points return identical values, which a
        # line search reads as a flat spot and probes until it gives up; the
        # tolerance below is well inside the inner solve's own accuracy.
        key = tuple(np.round(np.clip(p, lo, hi), 12))
        if key in _pcache:
            return _pcache[key]

        C, dC, _ = cost_and_grad(np.asarray(key))
        D = float(objective(np.asarray(key)))
        dD = np.array([(float(objective(np.clip(np.asarray(key) + h * e,
                                                lo, hi))) - D) / h
                       for e in np.eye(len(p))])
        out = ((C - C_id) / C_rng, dC / C_rng,
               (D - D_id) / D_rng, dD / D_rng)
        _pcache[key] = out
        return out

    if debug_vec:
        print(f"[codesign] normalization: C_id {C_id:.4g} C_rng {C_rng:.4g}, "
              f"D_id {D_id:.4g} D_rng {D_rng:.4g}")
        print(f"[codesign] anchors: C in [{Cg.min():.4g}, {Cg.max():.4g}], "
              f"D in [{Dg.min():.4g}, {Dg.max():.4g}]")
        print(f"{'w':>6}{'design':>34}{'Chat':>10}{'Dhat':>10}"
              f"{'w1*Chat':>10}{'w2*Dhat':>10}{'nit':>5}{'ok':>4}")

    front = []
    p_cur = p0.copy()
    for w in weights:
        w1, w2 = 1.0 - float(w), float(w)

        if norm in ("l1", "l2"):
            def fg(p):
                Ch, gC, Dh, gD = parts(p)
                if norm == "l1":
                    return w1 * Ch + w2 * Dh, w1 * gC + w2 * gD
                return ((w1 * Ch) ** 2 + (w2 * Dh) ** 2,
                        2.0 * (w1 ** 2 * Ch * gC + w2 ** 2 * Dh * gD))

            r = minimize(fg, p0.copy(), jac=True, method="L-BFGS-B",
                         bounds=list(zip(lo, hi)),
                         options=dict(maxiter=max_outer, ftol=1e-12,
                                      gtol=1e-10))
            p_cur = np.clip(r.x, lo, hi)

        else:
            # Chebyshev in epigraph form: minimize t subject to t >= w1*Chat
            # and t >= w2*Dhat. The maximum of two terms has no gradient where
            # they cross, and that crossing is exactly where the optimum sits,
            # so a smoothed maximum either stalls a gradient method or moves
            # the answer. The epigraph is exact and leaves the nonsmoothness in
            # the constraints, where an SQP handles it.
            # Started from the nominal design at every weight rather than from
            # the previous answer. Continuation is usually the right thing, but
            # the designs here run to the corners of the box, and an SQP
            # started at a corner where the previous weight left it reports
            # optimality without moving: the balance it should be looking for
            # is a long way off and nothing local points towards it.
            p_start = p0.copy()
            Ch0, _, Dh0, _ = parts(p_start)
            x0 = np.r_[p_start, max(w1 * Ch0, w2 * Dh0)]

            def f_ep(x):
                return x[-1] + rho * float(np.sum(parts(x[:-1])[::2]))

            def g_ep(x):
                Ch, gC, Dh, gD = parts(x[:-1])
                return np.r_[rho * (gC + gD), 1.0]

            def c_ep(x):
                Ch, _, Dh, _ = parts(x[:-1])
                return np.array([x[-1] - w1 * Ch, x[-1] - w2 * Dh])

            def cj_ep(x):
                _, gC, _, gD = parts(x[:-1])
                return np.vstack([np.r_[-w1 * gC, 1.0],
                                  np.r_[-w2 * gD, 1.0]])

            # ftol matched to what the inner solve can actually deliver. The
            # design enters the constraints through a numerical optimal control
            # problem, so the constraint values carry that solve's residual;
            # asking the SQP to resolve past it makes it iterate against noise,
            # which shows up as tens of iterations after the answer has stopped
            # moving, and as convergence failures at the end of them.
            r = minimize(f_ep, x0, jac=g_ep, method="SLSQP",
                         bounds=list(zip(lo, hi)) + [(None, None)],
                         constraints=[dict(type="ineq", fun=c_ep, jac=cj_ep)],
                         options=dict(maxiter=max(max_outer, 100),
                                      ftol=1e-7))
            p_cur = np.clip(r.x[:-1], lo, hi)

        if debug_vec:
            Ch, _, Dh, _ = parts(p_cur)
            print(f"{float(w):>6.2f}"
                  f"{np.array2string(p_cur, precision=4, max_line_width=200):>34}"
                  f"{Ch:>10.4f}{Dh:>10.4f}{w1 * Ch:>10.4f}{w2 * Dh:>10.4f}"
                  f"{getattr(r, 'nit', -1):>5}{str(bool(r.success)):>6}")

        # Final design solved properly, so the reported cost does not carry
        # the working tolerance the search ran at:
        sysp = _PinnedSystem(fam, nx, nu, N, z0, dt, p_cur)
        U = np.asarray(lambda_shoot(sysp, zt, U0=_warm_for(p_cur),
                                    ftol=1e-10, max_it=800,
                                    cost=cost)).flatten()
        # The normalization travels with the front. Comparing against another
        # method means solving the same scalarization, and that cannot be done
        # without the ideal and nadir these weights are measured against.
        front.append(dict(weight=float(w), param=p_cur.copy(),
                          cost=_cost_val(cost, U),
                          objective=float(objective(p_cur)), control=U,
                          norm=dict(C_id=C_id, C_rng=C_rng,
                                    D_id=D_id, D_rng=D_rng)))

    # Same dominance filter the scalar path uses; a nonzero drop count means
    # an inner solve went wrong rather than that a design is genuinely worse:
    pareto = pareto_front(front)
    n_dropped = len(front) - len(pareto)
    if n_dropped:
        print(f"[codesign] dropped {n_dropped} dominated point(s) of "
              f"{len(front)}")

    print(f"[codesign] {len(pareto)} front point(s) from {len(weights)} "
          f"weights, {len(anchors)} anchors, "
          f"{stats['solves']} inner solves")

    pick = pareto[len(pareto) // 2]
    if plot:
        _plot_front(pareto, save, job)
    return pick["control"], pick["param"], pareto, front


def codesign(dynamics, nx, nu, N, z0, dt, target, param_name, objective,
             p0, p_bounds, weights=None, substeps=1, save="figures",
             job="codesign", plot=True, target_idx=None, norm="cheby",
             n_anchor=9, beta=100.0, rho=1e-3, debug=False, p_tol=1e-5,
             max_outer=40, jit=True, jit_flags="-O1",
             cache_dir=".grace_cache", filter_dominated=True, cost=None,
             cost_dp=None):

    # cost is the same tuple lambda_shoot takes, (f, dinv[, dprime[, grad]]).
    # Supplying the gradient is worth it here even without constraints: the
    # design gradient recovers the multiplier by least squares at the returned
    # control, and without a gradient it has to fall back on the costate stored
    # at the last inner step instead. cost_dp(U, p) is the explicit design
    # dependence of the running cost, and is only needed when the cost is
    # written in terms of the design as well as the control.
    p0_arr = np.atleast_1d(np.asarray(p0, float))
    n_param = p0_arr.size
    vector_design = n_param > 1

    # Build the parameter family once:
    fam = _build_param_family(dynamics, nx, nu, N, z0, dt, param_name, substeps,
                              jit=jit, target_idx=target_idx,
                              jit_flags=jit_flags, cache_dir=cache_dir,
                              n_param=n_param)
    tidx = fam["tidx"]

    # A vector p0 means several parameters are being sized together. The
    # balance condition and the bracketed root find behind the scalar path both
    # assume a single design number, so a vector design takes a gradient outer
    # solve instead. The inner problem, the family, and the costate are the
    # same in both cases.
    if vector_design:
        return _codesign_vector(fam, tidx, nx, nu, N, z0, dt, target,
                                objective, p0_arr, p_bounds, weights, norm,
                                rho, p_tol, max_outer, save, job,
                                plot, n_anchor, cost=cost, cost_dp=cost_dp)

    # Reduce the target to the constrained components:
    zt = np.asarray(target, float)
    if zt.size == nx:
        zt = zt[tidx]

    # Convex weights so w sweeps the front end to end:
    if weights is None:
        weights = np.linspace(0.0, 1.0, 9)

    # Compute nominal control at the baseline design parameter:
    sys_nom = _PinnedSystem(fam, nx, nu, N, z0, dt, p0)
    U_nominal = lambda_shoot(sys_nom, zt, U0=None, cost=cost)

    # === ANCHOR SWEEP ===
    # Direct sweep over the scalar design parameter -- this is the exact front,
    # and it also supplies the ideal point used to normalize the scalarizations:
    p_grid = np.linspace(p_bounds[0], p_bounds[1], n_anchor)
    sweep = []
    U_warm = U_nominal
    for pv in p_grid:
        sp = _PinnedSystem(fam, nx, nu, N, z0, dt, float(pv))
        U_warm = lambda_shoot(sp, zt, U0=U_warm, cost=cost)
        sweep.append(dict(param=float(pv), cost=_cost_val(cost, U_warm),
                          objective=float(objective(float(pv))),
                          control=np.array(U_warm, copy=True)))
    C_grid = np.array([s["cost"] for s in sweep])
    D_grid = np.array([s["objective"] for s in sweep])

    # Verify stored controls match their recorded costs (catches seed aliasing):
    C_check = np.array([_cost_val(cost, s["control"]) for s in sweep])
    if not np.allclose(C_check, C_grid, rtol=1e-8):
        print("[codesign] WARNING: sweep controls do not match recorded costs")
        print(f"[codesign]   recorded: {np.array2string(C_grid, precision=1)}")
        print(f"[codesign]   from U:   {np.array2string(C_check, precision=1)}")

    # === SEED IDEMPOTENCY CHECK ===
    # Re-solve each anchor from its own converged control -- cost must not move:
    if debug:
        print(f"[codesign] {'p':>8} {'seed':>10} {'reseed':>10} "
              f"{'cold':>10} {'err_re':>9} {'err_cold':>9}")
        for s in sweep:
            sp = _PinnedSystem(fam, nx, nu, N, z0, dt, s["param"])
            U_re = lambda_shoot(sp, zt, U0=np.array(s["control"], copy=True),
                                cost=cost)
            U_cold = lambda_shoot(sp, zt, U0=None, cost=cost)
            e_re = float(np.linalg.norm(sp.endpoint(U_re) - zt))
            e_cold = float(np.linalg.norm(sp.endpoint(U_cold) - zt))
            print(f"[codesign] {s['param']:8.3f} {s['cost']:10.1f} "
                  f"{_cost_val(cost, U_re):10.1f} "
                  f"{_cost_val(cost, U_cold):10.1f} "
                  f"{e_re:9.2e} {e_cold:9.2e}")

    # Ideal point, padded below the grid min so normalized objectives stay positive:
    C_id = float(C_grid.min()) - 0.01 * float(np.ptp(C_grid))
    D_id = float(D_grid.min()) - 0.01 * float(np.ptp(D_grid))

    # Nadir is the opposing objective at each single-objective anchor:
    C_rng = max(float(C_grid[int(np.argmin(D_grid))]) - C_id, 1e-12)
    D_rng = max(float(D_grid[int(np.argmin(C_grid))]) - D_id, 1e-12)

    # === SCALARIZATION ===
    # Return the scalarized design gradient from normalized objectives and sensitivities:
    def scalarize(Chat, Dhat, ghat_t, ghat_o, w1, w2):

        # L1 -- weighted sum, reaches the convex hull only:
        if norm == "l1":
            return w1 * ghat_t + w2 * ghat_o

        # L2 -- curvature-limited reach into concave regions:
        if norm == "l2":
            return 2.0 * w1 ** 2 * Chat * ghat_t + 2.0 * w2 ** 2 * Dhat * ghat_o

        # Linf (Chebyshev) -- exact subgradient of the weighted max:
        if beta is None:
            if w1 * Chat > w2 * Dhat:
                gmax = w1 * ghat_t
            else:
                gmax = w2 * ghat_o

        # Linf smoothed by log-sum-exp so the finite-difference Hessian survives:
        else:
            a = np.array([beta * w1 * Chat, beta * w2 * Dhat])
            s = np.exp(a - a.max())
            s = s / s.sum()
            gmax = s[0] * w1 * ghat_t + s[1] * w2 * ghat_o

        # Augment with a small L1 term to exclude weakly Pareto points:
        return gmax + rho * (w1 * ghat_t + w2 * ghat_o)

    # Trace the Pareto front over the design weights:
    front = []

    # Continuation state: p(w) is a smooth curve, so the previous two solved
    # points predict the next one and the corrector only has to refine locally:
    hist_w, hist_p = [], []
    gref = [max(abs(float(C_grid.max() - C_grid.min())) / C_rng, 1.0)]

    # Control carried across weights too, since continuation keeps designs close:
    U = None

    for w in weights:

        # Weights enter the scalarization as a convex pair:
        w1 = 1.0 - float(w)
        w2 = float(w)

        # Total design gradient at a design value:
        def design_grad(pv, U_warm):
            sp = _PinnedSystem(fam, nx, nu, N, z0, dt, float(pv))
            Uv = U_warm
            _, Cov = sp.endpoint_jac(Uv)

            # Design sensitivity of the achieved control cost, from the
            # costate the inner solve already produces:
            #
            #     dJ/dp = -lam' (de/dp) + dJ/dp|_explicit
            #
            # The multiplier is read off the cost's own stationarity condition,
            # grad_U J = Co' lam, so nothing here assumes the cost is quadratic
            # -- the quadratic case is just grad_U J = 2U.
            #
            # The minus belongs on the contraction, not on the multiplier.
            # Folding it into lam and then not applying it here returns the
            # gradient with the wrong sign, which a root find answers by
            # walking to whichever bound is worst:
            lamv = _costate_at(sp, Uv, Cov, cost)
            dgp = np.array(fam["F_dp"](Uv, float(pv))).flatten()
            gt = -float(lamv @ dgp)

            # Explicit design dependence of the running cost, when there is one:
            if cost_dp is not None:
                gt = gt + float(np.asarray(cost_dp(Uv, float(pv))).flatten()[0])
            eps_o = 1e-6 * max(abs(p_bounds[1] - p_bounds[0]), 1e-6)
            go = (objective(pv + eps_o) - objective(pv - eps_o)) / (2 * eps_o)

            # Range-normalize both objectives from the ideal point:
            Chat = (_cost_val(cost, Uv) - C_id) / C_rng
            Dhat = (float(objective(pv)) - D_id) / D_rng
            return scalarize(Chat, Dhat, gt / C_rng, go / D_rng, w1, w2)

        # === OUTER SOLVE ===
        # For Chebyshev the optimum of max(w1*Chat, w2*Dhat) over a monotone
        # front sits exactly where the two terms balance. That balance function
        # is smooth and monotone, unlike the gradient, which has a kink there --
        # so no log-sum-exp smoothing is needed and the root find is easy.
        def balance_of_p(pv):
            nonlocal U
            sp = _PinnedSystem(fam, nx, nu, N, z0, dt, float(pv))
            seed = U if U is not None else np.array(
                sweep[int(np.argmin(np.abs(p_grid - float(pv))))]["control"],
                copy=True)
            U = lambda_shoot(sp, zt, U0=seed, cost=cost)
            Chat = (_cost_val(cost, U) - C_id) / C_rng
            Dhat = (float(objective(pv)) - D_id) / D_rng
            # Oriented so it is positive when the design term dominates, which
            # matches the gradient's sign convention at the bounds: positive at
            # the lower bound means the optimum sits there.
            return w2 * Dhat - w1 * Chat

        # Evaluate the gradient at a design, carrying the control as a warm start:
        def g_of_p(pv):
            nonlocal U
            sp = _PinnedSystem(fam, nx, nu, N, z0, dt, float(pv))
            seed = U if U is not None else np.array(
                sweep[int(np.argmin(np.abs(p_grid - float(pv))))]["control"],
                copy=True)
            U = lambda_shoot(sp, zt, U0=seed, cost=cost)
            return design_grad(float(pv), U)

        # Chebyshev uses the balance condition; other norms need the gradient:
        use_balance = norm not in ("l1", "l2")
        root_fn = balance_of_p if use_balance else g_of_p

        # A gradient that never changes sign puts the optimum at a bound:
        p_scale = max(abs(p_bounds[1] - p_bounds[0]), 1e-9)
        xt = p_tol * p_scale

        # Predict this weight's design from the continuation history:
        p_pred = None
        if len(hist_p) >= 2 and abs(hist_w[-1] - hist_w[-2]) > 1e-12:
            slope = (hist_p[-1] - hist_p[-2]) / (hist_w[-1] - hist_w[-2])
            p_pred = hist_p[-1] + slope * (float(w) - hist_w[-1])
        elif len(hist_p) == 1:
            p_pred = hist_p[-1]
        else:
            p_pred = float(np.clip(p0, p_bounds[0], p_bounds[1]))

        # The balance function is monotone, so a bracket is guaranteed to work
        # and is far more robust than extrapolating with a secant:
        if use_balance:
            p_pred = None

        # Correct from the prediction with a secant iteration -- typically 3-4
        # evaluations, against ~12 for a bracket search over the full range:
        p_solved = None
        if p_pred is not None:
            pa = float(np.clip(p_pred, p_bounds[0], p_bounds[1]))
            pb_ = float(np.clip(pa + max(4.0 * xt, 1e-3 * p_scale),
                                p_bounds[0], p_bounds[1]))
            if abs(pb_ - pa) > 1e-14:
                fa, fb = root_fn(pa), root_fn(pb_)

                # Absolute reference scale from the bounds, not from residuals
                # at the prediction -- a good predictor makes those tiny:
                gscale = max(gref[0], abs(fa), abs(fb), 1e-300)
                for _ in range(8):
                    if abs(fb - fa) < 1e-300:
                        break
                    pn = pb_ - fb * (pb_ - pa) / (fb - fa)

                    # A step outside the design box means the optimum is on it:
                    if pn < p_bounds[0] or pn > p_bounds[1]:
                        pn = float(np.clip(pn, p_bounds[0], p_bounds[1]))
                        if abs(pn - pb_) < xt:
                            p_solved = pn
                            break
                    pa, fa = pb_, fb
                    pb_ = float(pn)
                    fb = root_fn(pb_)

                    # Converged on the step or on the residual:
                    if abs(pb_ - pa) < xt or abs(fb) < 1e-10 * gscale:
                        p_solved = float(pb_)
                        break

                # Reject a corrector that landed on a non-root:
                if p_solved is not None and abs(fb) > 1e-4 * gscale and \
                        p_bounds[0] + xt < p_solved < p_bounds[1] - xt:
                    p_solved = None

        # Fall back to the full range when the corrector did not settle it:
        if p_solved is None:
            ga = root_fn(p_bounds[0])
            gb = root_fn(p_bounds[1])

            # Absolute gradient scale, used to judge later correctors:
            gref[0] = max(gref[0], abs(ga), abs(gb))
            if ga > 0.0:
                p_solved = float(p_bounds[0])
            elif gb < 0.0:
                p_solved = float(p_bounds[1])
            elif ga <= 0.0 <= gb:
                p_solved = float(brentq(root_fn, p_bounds[0], p_bounds[1],
                                        xtol=xt, rtol=1e-12))
            else:

                # Same sign at both bounds means the root function is not
                # monotone -- scan for an interior sign change:
                grid = np.linspace(p_bounds[0], p_bounds[1], max(n_anchor, 5))
                vals = [ga] + [root_fn(x) for x in grid[1:-1]] + [gb]
                for i in range(len(grid) - 1):
                    if vals[i] == 0.0:
                        p_solved = float(grid[i])
                        break
                    if vals[i] * vals[i + 1] < 0.0:
                        p_solved = float(brentq(root_fn, grid[i], grid[i + 1],
                                                xtol=xt, rtol=1e-12))
                        break

                # No crossing anywhere: one term attains the max throughout, so
                # the optimum sits on whichever bound minimizes it:
                if p_solved is None:
                    p_solved = float(p_bounds[0] if ga > 0.0 else p_bounds[1])

        # Record for the next prediction:
        hist_w.append(float(w))
        hist_p.append(p_solved)

        # Final control at the located design:
        sysp = _PinnedSystem(fam, nx, nu, N, z0, dt, p_solved)
        U = lambda_shoot(sysp, zt, U0=U, cost=cost)

        # Record the front point at the design the control was solved for. The
        # normalization travels with it: a caller who wants to state a trade in
        # their own units, or to pose the same scalarization to another
        # optimizer, cannot do either without the ideal and nadir the weights
        # are measured against:
        front.append(dict(weight=float(w), param=p_solved,
                          cost=_cost_val(cost, U),
                          objective=float(objective(p_solved)), control=U,
                          norm=dict(C_id=C_id, C_rng=C_rng,
                                    D_id=D_id, D_rng=D_rng)))

    # Dominance filter. It exists to catch an inner solve that went wrong,
    # since a bad solve shows up as a point another weight already beats on
    # both objectives. It is optional because coincident answers are also
    # legitimate: where both objectives improve in the same direction every
    # weight returns the same design, and filtering then collapses a full
    # weight sweep to a single point and discards the trade being traced:
    if filter_dominated:
        pareto = pareto_front(front)
        n_dropped = len(front) - len(pareto)
        if n_dropped:
            print(f"[codesign] dropped {n_dropped} dominated point(s) of "
                  f"{len(front)}")
    else:
        pareto = front

    # Select from the filtered front by the middle weight (balanced trade):
    pick = pareto[len(pareto) // 2]

    # Plot the scalarized front against the exact sweep:
    if plot:
        _plot_front(pareto, save, job, sweep=pareto_front(sweep))

    # Return the selected control, its design, the front, and the exact sweep:
    return pick["control"], pick["param"], pareto, sweep

# Plot a Pareto front of control effort versus design objective:
def _plot_front(front, save, job, sweep=None):

    # Import plotting locally so the package does not require it at import time:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot control cost against design objective across the front:
    costs = [f["cost"] for f in front]
    objs = [f["objective"] for f in front]
    fig, ax = plt.subplots(figsize=(7, 5))

    # Overlay the exact parameter sweep as the reference front:
    if sweep is not None:
        ax.plot([s["objective"] for s in sweep], [s["cost"] for s in sweep],
                "s--", color="0.6", lw=1.2, ms=4, label="exact sweep", zorder=1)

    ax.plot(objs, costs, "o-", color="steelblue", lw=1.8, label="scalarized", zorder=2)
    for f in front:
        ax.annotate(f"{f['param']:.2f}", (f["objective"], f["cost"]), fontsize=8)
    ax.set_xlabel("design objective")
    ax.set_ylabel("control effort")
    ax.set_title(f"Pareto front ({job}): effort vs design objective")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    # Save the figure:
    fig.savefig(save, dpi=110, bbox_inches="tight")
    return save

def pareto_front(front):

    pareto = []

    for i, fi in enumerate(front):

        dominated = False

        for j, fj in enumerate(front):

            if i == j:
                continue

            better_or_equal = (
                fj["cost"] <= fi["cost"]
                and
                fj["objective"] <= fi["objective"]
            )

            strictly_better = (
                fj["cost"] < fi["cost"]
                or
                fj["objective"] < fi["objective"]
            )

            if better_or_equal and strictly_better:
                dominated = True
                break

        if not dominated:
            pareto.append(fi)

    return pareto