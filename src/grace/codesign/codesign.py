# Import packages and shooting:
import os
import numpy as np
from ..shooting.bounds import safe_solve
import casadi as ca

from ..shooting.lambda_simple import lambda_simple
from ..core.system import build as build_system


# Build a parameterized system family and its parameter-sensitivity rollout:
def _build_param_family(dynamics, nx, nu, N, z0, dt, param_name, substeps=1, jit=True,
                        target_idx=None):

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

    # Build the full rollout with a compiled scan (mapaccum) rather than an
    # unrolled Python loop -- the unrolled graph is what made high-dimensional
    # nonlinear rollouts (e.g. the 6DOF aircraft) slow to evaluate:
    U = ca.MX.sym("U", N * nu)
    Useq = ca.reshape(U, nu, N)
    acc = step.mapaccum("roll", N)
    Zc = acc(ca.DM(z0), Useq, ca.repmat(p, 1, N))
    Zmat = ca.horzcat(ca.DM(z0), Zc).T

    # Constrain only the requested endpoint components (default: all states).
    # For high-dimensional systems like the 6DOF aircraft, targeting every state
    # over-constrains the shoot; a maneuver targets the states that define it
    # (e.g. attitude and rates for a roll) and lets the rest settle freely:
    tidx = list(range(nx)) if target_idx is None else list(target_idx)
    gend = Zmat[-1, tidx].T

    # Compile the endpoint, its control Jacobian, and its parameter sensitivity,
    # JIT-compiled so the nested shoot runs in milliseconds:
    opts = {"jit": jit, "compiler": "shell",
            "jit_options": {"flags": ["-O2"]}} if jit else {}
    F_end = ca.Function("F_end_p", [U, p], [gend], opts)
    F_dU = ca.Function("F_dU_p", [U, p], [ca.jacobian(gend, U)], opts)
    F_dp = ca.Function("F_dp_p", [U, p], [ca.jacobian(gend, p)], opts)
    F_roll = ca.Function("F_roll_p", [U, p], [Zmat], opts)

    # Return the compiled parameter family (with the targeted endpoint indices):
    return dict(step=step, F_end=F_end, F_dU=F_dU, F_dp=F_dp, F_roll=F_roll, tidx=tidx)


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

    # Roll out at the pinned parameter:
    def rollout(self, U):
        return np.array(self.fam["F_roll"](np.asarray(U).flatten(), self.pval))

    # Endpoint and control Jacobian at the pinned parameter:
    def endpoint_jac(self, U):
        g = np.array(self.fam["F_end"](np.asarray(U).flatten(), self.pval)).flatten()
        J = np.array(self.fam["F_dU"](np.asarray(U).flatten(), self.pval))
        return g, J

    def endpoint(self, U):
        return np.array(self.fam["F_end"](np.asarray(U).flatten(), self.pval)).flatten()

    def target(self, z_target):
        zt = np.asarray(z_target, float)
        return zt[self.tidx] if len(zt) == self.nx else zt


# Run codesign over a named design parameter:
def codesign(dynamics, nx, nu, N, z0, dt, target, param_name, objective,
             p0, p_bounds, weights=None, substeps=1, figures_dir="figures",
             job="codesign", plot=True, target_idx=None):

    # dynamics here takes (x, u, p): the design parameter appears explicitly.
    # objective(p) is the design cost to trade against control effort.

    # Build the parameter family once (optionally targeting a subset of states):
    fam = _build_param_family(dynamics, nx, nu, N, z0, dt, param_name, substeps,
                              target_idx=target_idx)
    tidx = fam["tidx"]

    # Reduce the target to the constrained components:
    zt = np.asarray(target, float)
    if zt.size == nx:
        zt = zt[tidx]

    # Default the weight sweep if none was given:
    if weights is None:
        weights = np.linspace(0.0, 2.0, 9)

    # Trace the Pareto front over the design weights:
    front = []
    for w in weights:

        # Alternate inner control optimization with outer design descent:
        p = float(p0)
        U = None
        prev_p = None
        for _ in range(40):

            # Inner solve -- optimal control at the current design:
            sysp = _PinnedSystem(fam, nx, nu, N, z0, dt, p)
            U = lambda_simple(sysp, zt, U0=U)

            # Total design gradient at p: control-cost sensitivity (envelope theorem)
            # plus the weighted design objective.  Evaluated by a helper so we can also
            # finite-difference it for the design curvature (a damped Newton step on the
            # scalar design converges to the true interior optimum instead of railing to
            # a bound, which is what a plain gradient step does):
            def design_grad(pv):
                sp = _PinnedSystem(fam, nx, nu, N, z0, dt, pv)
                Uv = lambda_simple(sp, zt, U0=U)
                _, Cov = sp.endpoint_jac(Uv)
                Wv = Cov @ Cov.T + 1e-6 * np.eye(sp.m)
                lamv = safe_solve(Wv, -Cov @ (2 * Uv))
                dgp = np.array(fam["F_dp"](Uv, pv)).flatten()
                gt = float(lamv @ dgp)
                eps_o = 1e-6 * max(abs(p_bounds[1] - p_bounds[0]), 1e-6)
                go = (objective(pv + eps_o) - objective(pv - eps_o)) / (2 * eps_o)
                return gt + w * go

            grad = design_grad(p)

            # Design curvature by central finite difference (cheap: scalar parameter):
            dp = 1e-3 * max(abs(p_bounds[1] - p_bounds[0]), 1e-6)
            hess = (design_grad(p + dp) - design_grad(p - dp)) / (2 * dp)

            # Damped Newton step (Levenberg): use curvature when positive, else fall back
            # to a scaled gradient step.  Keeps the update stable and interior:
            p_scale = max(abs(p_bounds[1] - p_bounds[0]), 1e-9)
            if hess > 1e-9:
                step = -grad / (hess + 1e-6)
            else:
                step = -0.1 * p_scale * np.sign(grad)
            # limit the step to a fraction of the parameter range for stability:
            step = float(np.clip(step, -0.5 * p_scale, 0.5 * p_scale))
            p = float(np.clip(p + step, p_bounds[0], p_bounds[1]))

            # Stop when the design settles:
            if prev_p is not None and abs(p - prev_p) < 1e-5 * p_scale:
                break
            prev_p = p

        # Record the front point:
        front.append(dict(weight=float(w), param=p, cost=float(U @ U),
                          objective=float(objective(p)), control=U))

    # Select the returned control by the middle weight (balanced trade):
    pick = front[len(front) // 2]

    # Plot the Pareto front to the figures directory:
    if plot:
        _plot_front(front, figures_dir, job)

    # Return the selected control and the full front:
    return pick["control"], pick["param"], front


# Plot a Pareto front of control effort versus design objective:
def _plot_front(front, figures_dir, job):

    # Import plotting locally so the package does not require it at import time:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ensure the figures directory exists:
    os.makedirs(figures_dir, exist_ok=True)

    # Plot control cost against design objective across the front:
    costs = [f["cost"] for f in front]
    objs = [f["objective"] for f in front]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(objs, costs, "o-", color="steelblue", lw=1.8)
    for f in front:
        ax.annotate(f"{f['param']:.2f}", (f["objective"], f["cost"]), fontsize=8)
    ax.set_xlabel("design objective")
    ax.set_ylabel("control effort")
    ax.set_title(f"Pareto front ({job}): effort vs design objective")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    # Save the figure:
    path = os.path.join(figures_dir, f"{job}_pareto.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path