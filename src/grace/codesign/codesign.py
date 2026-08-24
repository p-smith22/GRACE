# Import packages and shooting:
import os
import hashlib
import numpy as np
from ..shooting.bounds import safe_solve
import casadi as ca
from scipy.optimize import brentq
from ..shooting.lambda_shoot import lambda_shoot

# Build a parameterized system family and its parameter-sensitivity rollout:
def _build_param_family(dynamics, nx, nu, N, z0, dt, param_name, substeps=1, jit=True,
                        target_idx=None, jit_flags="-O1", cache_dir=".grace_cache"):

    # Symbolic state, control, and the named design parameter:
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    p = ca.MX.sym(param_name, 1)

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
        sig = f"{str(gend)}|{str(Zmat)}|{jit_flags}"
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
        self.pval = pval
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
         filter_dominated=True):

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
        U = lambda_shoot(sp, zt, U0=U)
        front.append(dict(param=float(pv), cost=float(U @ U),
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
def codesign(dynamics, nx, nu, N, z0, dt, target, param_name, objective,
             p0, p_bounds, weights=None, substeps=1, save="figures",
             job="codesign", plot=True, target_idx=None, norm="cheby",
             n_anchor=9, beta=100.0, rho=1e-3, debug=False, p_tol=1e-5,
             max_outer=40, jit=True, jit_flags="-O1",
             cache_dir=".grace_cache"):

    # Build the parameter family once:
    fam = _build_param_family(dynamics, nx, nu, N, z0, dt, param_name, substeps,
                              jit=jit, target_idx=target_idx,
                              jit_flags=jit_flags, cache_dir=cache_dir)
    tidx = fam["tidx"]

    # Reduce the target to the constrained components:
    zt = np.asarray(target, float)
    if zt.size == nx:
        zt = zt[tidx]

    # Convex weights so w sweeps the front end to end:
    if weights is None:
        weights = np.linspace(0.0, 1.0, 9)

    # Compute nominal control at the baseline design parameter:
    sys_nom = _PinnedSystem(fam, nx, nu, N, z0, dt, p0)
    U_nominal = lambda_shoot(sys_nom, zt, U0=None)

    # === ANCHOR SWEEP ===
    # Direct sweep over the scalar design parameter -- this is the exact front,
    # and it also supplies the ideal point used to normalize the scalarizations:
    p_grid = np.linspace(p_bounds[0], p_bounds[1], n_anchor)
    sweep = []
    U_warm = U_nominal
    for pv in p_grid:
        sp = _PinnedSystem(fam, nx, nu, N, z0, dt, float(pv))
        U_warm = lambda_shoot(sp, zt, U0=U_warm)
        sweep.append(dict(param=float(pv), cost=float(U_warm @ U_warm),
                          objective=float(objective(float(pv))),
                          control=np.array(U_warm, copy=True)))
    C_grid = np.array([s["cost"] for s in sweep])
    D_grid = np.array([s["objective"] for s in sweep])

    # Verify stored controls match their recorded costs (catches seed aliasing):
    C_check = np.array([float(s["control"] @ s["control"]) for s in sweep])
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
            U_re = lambda_shoot(sp, zt, U0=np.array(s["control"], copy=True))
            U_cold = lambda_shoot(sp, zt, U0=None)
            e_re = float(np.linalg.norm(sp.endpoint(U_re) - zt))
            e_cold = float(np.linalg.norm(sp.endpoint(U_cold) - zt))
            print(f"[codesign] {s['param']:8.3f} {s['cost']:10.1f} "
                  f"{float(U_re @ U_re):10.1f} {float(U_cold @ U_cold):10.1f} "
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
            Wv = Cov @ Cov.T + 1e-6 * np.eye(sp.m)
            lamv = safe_solve(Wv, -Cov @ (2 * Uv))
            dgp = np.array(fam["F_dp"](Uv, float(pv))).flatten()
            gt = float(lamv @ dgp)
            eps_o = 1e-6 * max(abs(p_bounds[1] - p_bounds[0]), 1e-6)
            go = (objective(pv + eps_o) - objective(pv - eps_o)) / (2 * eps_o)

            # Range-normalize both objectives from the ideal point:
            Chat = (float(Uv @ Uv) - C_id) / C_rng
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
            U = lambda_shoot(sp, zt, U0=seed)
            Chat = (float(U @ U) - C_id) / C_rng
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
            U = lambda_shoot(sp, zt, U0=seed)
            return design_grad(float(pv), U)

        # Chebyshev uses the balance condition; other norms need the gradient:
        use_balance = norm not in ("l1", "l2")
        root_fn = balance_of_p if use_balance else g_of_p

        # The balance function is monotone, so a bracket is guaranteed to work
        # and is far more robust than extrapolating with a secant:
        if use_balance:
            p_pred = None

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
            else:
                p_solved = float(brentq(root_fn, p_bounds[0], p_bounds[1],
                                        xtol=xt, rtol=1e-12))

        # Record for the next prediction:
        hist_w.append(float(w))
        hist_p.append(p_solved)

        # Final control at the located design:
        sysp = _PinnedSystem(fam, nx, nu, N, z0, dt, p_solved)
        U = lambda_shoot(sysp, zt, U0=U)

        # Record the front point at the design the control was solved for:
        front.append(dict(weight=float(w), param=p_solved, cost=float(U @ U),
                          objective=float(objective(p_solved)), control=U))


    # Filter dominated points -- a nonzero drop count signals inner-solve trouble:
    pareto = pareto_front(front)
    n_dropped = len(front) - len(pareto)
    if n_dropped:
        print(f"[codesign] dropped {n_dropped} dominated point(s) of {len(front)}")

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