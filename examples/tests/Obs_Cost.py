# Import packages:
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, nnls
import grace

# === PLANT ===
# Same lateral-directional model, with downtrack distance carried as a state.
# The keep-out is a circle in the ground plane, so a node-wise constraint needs
# both coordinates in z; x integrates at a constant rate and is left free at
# the endpoint, so it adds a coordinate without adding a degree of freedom:
IXX, IZZ = 4.0, 6.5
L_A, L_W, L_R = 0.90, 0.30, 0.05
N_A, N_W, N_R = -0.10, 0.55, 0.45
D_P, D_R = 1.6, 1.1
Y_B, Y_W, Y_R = -0.20, -0.05, 0.08
V_TAS, G_ACC = 45.0, 9.81
L_REF = 10.0

def dynamics(z, u):
    phi, p, psi, r, beta = z[0], z[1], z[2], z[3], z[4]
    da, dw, dr = u[0], u[1], u[2]
    return ca.vertcat(
        p,
        (L_A * da + L_W * dw + L_R * dr
         - D_P * p * ca.sqrt(p ** 2 + 1e-6)) / IXX,
        r,
        (N_A * da + N_W * dw + N_R * dr
         - D_R * r * ca.sqrt(r ** 2 + 1e-6)) / IZZ,
        (G_ACC / V_TAS) * phi - r + Y_B * beta + Y_W * dw + Y_R * dr,
        V_TAS * (psi + beta) / L_REF,
        V_TAS / L_REF)

nx, nu = 7, 3
NAMES = ("aileron", "winglet", "rudder")

# The maneuver is a deviation and return: the aircraft leaves the centerline,
# clears a keep-out sitting on it, and comes back wings level, on heading and
# coordinated. With the target back at the centerline the whole excursion is
# bought by the obstacle, so the constraint is doing all the work and nothing
# is inherited from a commanded offset:
T_MAN = 8.0
N, dt = 40, T_MAN / 40
Y_TARGET_M = 0.0
z0 = np.zeros(nx)
target = np.zeros(nx)
target[5] = Y_TARGET_M / L_REF
TIDX = [0, 1, 2, 3, 4, 5]

# === KEEP-OUT ===
# On the centerline at midcourse, so the straight path runs through the middle
# of it. h is written as an area, positive inside, which is the form the
# solver's saddle escape reads as a distance still to travel:
X_OBS, Y_OBS, R_OBS = 18.0, 0.0, 0.90

def keep_out(z, u):
    return R_OBS ** 2 - ((z[6] - X_OBS) ** 2 + (z[5] - Y_OBS) ** 2)

# === COST ===
# Hinge moment with downwash interference, unchanged from the lane change:
RHO = 1.225
Q_DYN = 0.5 * RHO * V_TAS ** 2
S_SURF = np.array([0.32, 0.18, 0.22])
C_SURF = np.array([0.14, 0.11, 0.16])
CH_ALPHA, CH_DELTA = -0.28, -0.52
EPS_W_ON_A, EPS_A_ON_W = 1.67, 0.75
ALPHA = EPS_W_ON_A * CH_ALPHA / CH_DELTA
KAPPA = (EPS_A_ON_W / EPS_W_ON_A) ** 2
B_HNG = Q_DYN * S_SURF * C_SURF * abs(CH_DELTA)
A_BRK = np.array([1.85, 0.55, 1.05])
EPS_ABS = 0.02
THRESH = A_BRK / (2.0 * B_HNG)

def kmat(alpha=ALPHA):
    return np.array([[1.0,           -alpha, 0.0],
                     [-KAPPA * alpha, 1.0,   0.0],
                     [0.0,            0.0,   1.0]])

def phi_steps(V):
    c = A_BRK * (np.sqrt(V ** 2 + EPS_ABS ** 2) - EPS_ABS) + B_HNG * V ** 2
    return np.maximum(c, 0.0)

def phi_grad(V):
    return A_BRK * V / np.sqrt(V ** 2 + EPS_ABS ** 2) + 2.0 * B_HNG * V

def phi_hess(V):
    return (A_BRK * EPS_ABS ** 2 / (V ** 2 + EPS_ABS ** 2) ** 1.5
            + 2.0 * B_HNG)

def phi_inv(W, iters=64):
    hi = A_BRK / EPS_ABS + 2.0 * B_HNG
    lo = 2.0 * B_HNG
    aw = np.abs(W)
    x0, x1 = aw / hi, aw / lo
    for _ in range(iters):
        xm = 0.5 * (x0 + x1)
        small = phi_grad(xm) < aw
        x0 = np.where(small, xm, x0)
        x1 = np.where(small, x1, xm)
    return np.sign(W) * 0.5 * (x0 + x1)

def step_costs(U):
    return phi_steps(np.asarray(U).reshape(-1, nu) @ kmat().T)

def coupled():
    K = kmat()
    KI = np.linalg.inv(K)

    def f(U):
        return float(np.sum(phi_steps(np.asarray(U).reshape(-1, nu) @ K.T)))

    def dinv(S):
        return (phi_inv(np.asarray(S).reshape(-1, nu) @ KI) @ KI.T).ravel()

    # M = (grad^2 g)^-1, one nu-by-nu block per node, which is the form the
    # constrained metric expects:
    def blocks(S):
        V = phi_inv(np.asarray(S).reshape(-1, nu) @ KI)
        return np.einsum("ij,kj,lj->kil", KI, 1.0 / phi_hess(V), KI)

    # The gradient the constrained path reads at damped iterates, where the
    # control is no longer the dual map's image:
    def grad(U):
        return (phi_grad(np.asarray(U).reshape(-1, nu) @ K.T) @ K).ravel()

    return f, dinv, blocks, grad

# === BASELINE KEEP-OUT JACOBIAN ===
# The baseline gets the keep-out rows and their exact Jacobian, built here from
# the same RK4 the solver integrates with. Without it SLSQP differences the
# constraint by hand, which is 121 rollouts per Jacobian at this problem size
# and swamps every other number in the comparison. GRACE's constraint Jacobian
# is analytic, so the baseline's has to be as well or the two are not being
# asked to do the same work:
def build_keepout():
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    k1 = dynamics(z, u)
    k2 = dynamics(z + 0.5 * dt * k1, u)
    k3 = dynamics(z + 0.5 * dt * k2, u)
    k4 = dynamics(z + dt * k3, u)
    step = ca.Function("step", [z, u],
                       [z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])
    U = ca.MX.sym("U", N * nu)
    Zc = step.mapaccum("roll", N)(ca.DM(z0), ca.reshape(U, nu, N))
    Z = ca.horzcat(ca.DM(z0), Zc).T

    # Positive outside the circle, which is the sign an inequality row takes:
    h = ((Z[:, 6] - X_OBS) ** 2 + (Z[:, 5] - Y_OBS) ** 2 - R_OBS ** 2)
    return (ca.Function("h", [U], [h]),
            ca.Function("hj", [U], [ca.jacobian(h, U)]))

# === INSTRUMENTATION ===
# Wall time alone confounds the solver with the cost of one rollout, so the
# calls each method makes through the plant are counted alongside it. row_jac
# is included because that is where GRACE's constraint Jacobian comes from;
# leaving it out credits GRACE with work the baseline is charged for:
def instrument(system):
    if getattr(system, "_counted", False):
        return system._counts
    roll, endp, ejac = system.rollout, system.endpoint, system.endpoint_jac
    rjac = getattr(system, "row_jac", None)
    counts = dict(roll=0, end=0, jac=0, rjac=0, h=0, hjac=0)

    def rollout(U):
        counts["roll"] += 1
        return roll(U)

    def endpoint(U):
        counts["end"] += 1
        return endp(U)

    def endpoint_jac(U):
        counts["jac"] += 1
        return ejac(U)

    system.rollout = rollout
    system.endpoint = endpoint
    system.endpoint_jac = endpoint_jac
    if rjac is not None:

        def row_jac(U, states):
            counts["rjac"] += 1
            return rjac(U, states)

        system.row_jac = row_jac
    system._counts = counts
    system._counted = True
    return counts

def snap(counts):
    return dict(counts)

def reset(counts):
    for k in counts:
        counts[k] = 0

def total_evals(c):

    # Every entry is one pass over the horizon, forward or reverse, so they add:
    return c["roll"] + c["end"] + c["jac"] + c["rjac"] + c["h"] + c["hjac"]

# === CHECKS ===
def clearance(Z):
    d = np.sqrt((Z[:, 6] - X_OBS) ** 2 + (Z[:, 5] - Y_OBS) ** 2)
    return float(d.min() - R_OBS)

def stationarity(system, U, grad, Jh=None, act=None):

    # Stationarity of the full Lagrangian, grad g + Co' lam + Jh' eta. The
    # endpoint multiplier is free, so it is split into a positive and a
    # negative part; h is positive inside the keep-out, so the feasible set is
    # h <= 0 and its multiplier enters with a plus sign and is constrained
    # nonnegative. Signing that the other way asks for a nonpositive
    # multiplier, which no correct solution can supply, and the residual then
    # reads as a failure at a converged point:
    _, Co = system.endpoint_jac(U)
    gr = np.asarray(grad(U)).flatten()
    m = Co.shape[0]
    if Jh is None or act is None or len(act) == 0:
        lam, *_ = np.linalg.lstsq(Co.T, -gr, rcond=None)
        return (float(np.linalg.norm(gr + Co.T @ lam)
                      / max(np.linalg.norm(gr), 1e-30)), lam)
    A = np.vstack([Co, Jh[act]])
    M = np.hstack([A.T[:, :m], -A.T[:, :m], A.T[:, m:]])
    w, res = nnls(M, -gr)
    lam = np.r_[w[:m] - w[m:2 * m], w[2 * m:]]
    return float(res / max(np.linalg.norm(gr), 1e-30)), lam

def endpoint_err(system, U, tgt):
    return float(np.linalg.norm(np.asarray(system.endpoint(U))
                                - system.target(tgt)))

# === BASELINE ===
# Seed the baseline off the symmetric saddle. With the keep-out centred on the
# straight path, dh/dy vanishes at U = 0 and a gradient method has no direction
# to move in. GRACE breaks that symmetry internally, so the baseline is given
# the same information rather than left on the saddle -- an aileron doublet,
# the simplest input that bulges the track to one side and returns:
def nlp_seed(scale=0.15):
    tt = np.arange(N) * dt
    U0 = np.zeros((N, nu))
    U0[:, 0] = scale * np.sin(2.0 * np.pi * tt / T_MAN)
    return U0.ravel()

def nlp_solve(system, tgt, f, grad, f_h, f_hj, counts, obstacle=True,
              maxiter=500, U0=None):
    zt = system.target(tgt)

    # Endpoint rows, with the analytic Jacobian the same rollout produces:
    def eq_fun(v):
        return np.asarray(system.endpoint(v)) - zt

    def eq_jac(v):
        return np.asarray(system.endpoint_jac(v)[1])

    constraints = [dict(type="eq", fun=eq_fun, jac=eq_jac)]

    # Keep-out rows, analytic as well, so neither block is differenced:
    if obstacle:

        def h_fun(v):
            counts["h"] += 1
            return np.asarray(f_h(v)).flatten()

        def h_jac(v):
            counts["hjac"] += 1
            return np.asarray(f_hj(v))

        constraints.append(dict(type="ineq", fun=h_fun, jac=h_jac))

    return minimize(f, nlp_seed() if U0 is None else U0, jac=grad,
                    method="SLSQP", constraints=constraints,
                    options=dict(maxiter=maxiter, ftol=1e-9))

# === RUN ===
if __name__ == "__main__":
    system = grace.build_cached(dynamics, nx=nx, nu=nu, N=N, z0=z0, dt=dt,
                                target_idx=TIDX, job="lanechange_obs")
    counts = instrument(system)
    engine = grace.GRACE(system)
    shoot = engine.shooting.lambda_shoot
    f_g, dinv_g, blk_g, grad_g = coupled()
    f_h, f_hj = build_keepout()

    # Warm the compiled functions, so the first timed solve is not paying for
    # a lazy first call the second one never sees:
    _ = f_h(nlp_seed())
    _ = f_hj(nlp_seed())
    _ = system.endpoint_jac(nlp_seed())
    reset(counts)

    print(f"deviate and return in {T_MAN:.0f} s at {V_TAS:.0f} m/s, "
          f"keep-out at ({L_REF * X_OBS:.0f}, {L_REF * Y_OBS:.0f}) m, "
          f"radius {L_REF * R_OBS:.0f} m")
    print(f"breakout threshold "
          f"{np.array2string(np.rad2deg(THRESH), precision=2)} deg")
    print(f"hinge Hessian per channel: "
          f"{np.array2string(2.0 * B_HNG, precision=1)} at large deflection to "
          f"{np.array2string(A_BRK / EPS_ABS + 2.0 * B_HNG, precision=1)} "
          f"at the breakout, N m/rad^2\n")

    # Active rows and the certificate they belong in:
    def certify(U, grad):
        h0 = -np.asarray(f_h(U)).flatten()
        Jh = -np.asarray(f_hj(U))
        act = np.where(h0 > -1e-4)[0]
        st, _ = stationarity(system, U, grad, Jh, act)
        return st, act.size

    runs = {}
    for label, kw, fn, gr in (
            ("quadratic", {}, lambda V: 0.5 * float(V @ V), lambda V: V),
            ("coupled g", dict(cost=(f_g, dinv_g, blk_g, grad_g)),
             f_g, grad_g)):

        # GRACE, with the keep-out:
        reset(counts)
        t0 = time.perf_counter()
        U = np.asarray(shoot(target, constraints=[keep_out], **kw)).flatten()
        t_grace = time.perf_counter() - t0
        c_grace = snap(counts)
        infeas = bool(getattr(system, "_infeasible", False))
        Z = np.asarray(system.rollout(U))

        # Cold-started SLSQP, analytic Jacobians on both constraint blocks:
        reset(counts)
        t0 = time.perf_counter()
        res = nlp_solve(system, target, fn, gr, f_h, f_hj, counts)
        t_nlp = time.perf_counter() - t0
        c_nlp = snap(counts)
        U_nlp = np.asarray(res.x).flatten()
        Z_nlp = np.asarray(system.rollout(U_nlp))

        # Certificates, taken outside the timed regions:
        st_grace, n_act = certify(U, gr)
        st_nlp, _ = certify(U_nlp, gr)

        runs[label] = dict(
            U=U, t_grace=t_grace, Z=Z, C=step_costs(U), c_grace=c_grace,
            infeas=infeas, stat=st_grace, n_act=n_act,
            err=endpoint_err(system, U, target), clear=L_REF * clearance(Z),
            U_nlp=U_nlp, Z_nlp=Z_nlp, C_nlp=step_costs(U_nlp),
            t_nlp=t_nlp, c_nlp=c_nlp, nlp_ok=bool(res.success),
            its_nlp=res.nit, nlp_msg=str(res.message), stat_nlp=st_nlp,
            nlp_end=endpoint_err(system, U_nlp, target),
            nlp_clear=L_REF * clearance(Z_nlp),
            dobj=abs(fn(U) - fn(U_nlp)) / max(abs(fn(U_nlp)), 1e-30),
            dU=float(np.abs(U - U_nlp).max()))

    # Unconstrained coupled run, so the price of the keep-out is separable
    # from the price of the cost:
    reset(counts)
    U_free = np.asarray(shoot(target, cost=(f_g, dinv_g, blk_g))).flatten()
    Z_free = np.asarray(system.rollout(U_free))

    print("GRACE, keep-out active")
    print(f"{'':<12}{'g(U)':>10}{'||U||^2':>10}{'clear':>9}{'act':>5}"
          f"{'stat':>10}{'endpt':>10}{'s':>8}{'roll':>7}{'jac':>6}"
          f"{'rjac':>6}{'passes':>8}")
    for label, d in runs.items():
        c = d["c_grace"]
        print(f"{label:<12}{f_g(d['U']):>10.5g}"
              f"{float(d['U'] @ d['U']):>10.5g}{d['clear']:>8.2f}m"
              f"{d['n_act']:>5d}{d['stat']:>10.1e}{d['err']:>10.1e}"
              f"{d['t_grace']:>8.3f}{c['roll']:>7d}{c['jac']:>6d}"
              f"{c['rjac']:>6d}{total_evals(c):>8d}"
              f"{'  INFEASIBLE' if d['infeas'] else ''}")
    print(f"{'unconstrained':<12}{f_g(U_free):>10.5g}"
          f"{float(U_free @ U_free):>10.5g}"
          f"{L_REF * clearance(Z_free):>8.2f}m")
    print(f"keep-out costs {f_g(runs['coupled g']['U']) - f_g(U_free):.3f} "
          f"N m of hinge work over the unconstrained solution\n")

    print("cold-start SLSQP, analytic Jacobians on both constraint blocks")
    print(f"{'':<12}{'s':>8}{'it':>5}{'end':>6}{'jac':>6}{'h':>6}{'hjac':>6}"
          f"{'passes':>8}{'d obj':>10}{'max dU':>10}{'stat':>10}{'clear':>9}")
    for label, d in runs.items():
        c = d["c_nlp"]
        print(f"{label:<12}{d['t_nlp']:>8.3f}{d['its_nlp']:>5d}"
              f"{c['end']:>6d}{c['jac']:>6d}{c['h']:>6d}{c['hjac']:>6d}"
              f"{total_evals(c):>8d}{d['dobj']:>10.1e}{d['dU']:>10.1e}"
              f"{d['stat_nlp']:>10.1e}{d['nlp_clear']:>8.2f}m")
        if not d["nlp_ok"]:
            print(f"  {d['nlp_msg']}")

    print(f"\n{'':<12}{'wall time':>12}{'horizon passes':>18}")
    for label, d in runs.items():
        sp_t = d["t_nlp"] / max(d["t_grace"], 1e-12)
        sp_e = total_evals(d["c_nlp"]) / max(total_evals(d["c_grace"]), 1)
        print(f"{label:<12}{sp_t:>11.1f}x{sp_e:>17.1f}x")

    Cq, Cc = runs["quadratic"]["C"], runs["coupled g"]["C"]
    print(f"\nper-surface cost under g, both constrained")
    print(f"{'':<10}{'quadratic':>12}{'coupled g':>12}{'change':>10}")
    for j, nm in enumerate(NAMES):
        sq, sc = Cq[:, j].sum(), Cc[:, j].sum()
        print(f"{nm:<10}{sq:>12.4f}{sc:>12.4f}"
              f"{(sc / sq - 1.0) * 100:>9.1f}%")
    print(f"{'total':<10}{Cq.sum():>12.4f}{Cc.sum():>12.4f}"
          f"{(Cc.sum() / Cq.sum() - 1.0) * 100:>9.1f}%")

    # === PLOT ===
    t = np.arange(N) * dt
    cols = {"quadratic": "0.45", "coupled g": "crimson"}

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.2))
    ax = ax.ravel()

    # Trajectory:
    th = np.linspace(0.0, 2.0 * np.pi, 200)
    ax[0].fill(L_REF * (X_OBS + R_OBS * np.cos(th)),
               L_REF * (Y_OBS + R_OBS * np.sin(th)),
               color="firebrick", alpha=0.20, label="keep-out")
    ax[0].plot(L_REF * Z_free[:, 6], L_REF * Z_free[:, 5], ":",
               color="crimson", lw=1.5, alpha=0.75,
               label="coupled g, no keep-out")
    for label, d in runs.items():
        ax[0].plot(L_REF * d["Z"][:, 6], L_REF * d["Z"][:, 5], "-",
                   color=cols[label], lw=2.2,
                   label=f"{label}  (g = {f_g(d['U']):.3f})")
        ax[0].plot(L_REF * d["Z_nlp"][::4, 6], L_REF * d["Z_nlp"][::4, 5],
                   "o", color=cols[label], ms=3.5, mfc="none", lw=0)
    ax[0].plot(L_REF * runs["coupled g"]["Z"][-1, 6], Y_TARGET_M, "*",
               color="k", ms=13, label="commanded")
    ax[0].set_xlabel("Downtrack [m]")
    ax[0].set_ylabel("Lateral offset [m]")
    ax[0].set_title("Trajectory")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # Total cost per step:
    for label, d in runs.items():
        s = d["C"].sum(axis=1)
        ax[1].plot(t, s, "-", color=cols[label], lw=2.2,
                   label=f"{label}  (total {s.sum():.2f})")
    ax[1].fill_between(t, Cc.sum(axis=1), Cq.sum(axis=1),
                       where=Cq.sum(axis=1) >= Cc.sum(axis=1),
                       color="crimson", alpha=0.12)
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Control Cost [N m]")
    ax[1].set_title("Total Cost")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # Per-surface cost, with the deflection that produced it:
    for j in range(nu):
        a_ = ax[2 + j]
        for label, d in runs.items():
            a_.plot(t, d["C"][:, j], "-", color=cols[label], lw=2.2,
                    label=f"{label}  ({d['C'][:, j].sum():.3f})")
            a_.plot(t[::4], d["C_nlp"][::4, j], "o", color=cols[label],
                    ms=3.5, mfc="none", lw=0)
        a_.fill_between(t, Cc[:, j], Cq[:, j], where=Cq[:, j] >= Cc[:, j],
                        color="crimson", alpha=0.12)
        a_.set_ylim(bottom=0.0)
        a_.set_xlabel("Time [s]")
        a_.set_ylabel("Control Cost [N m]")
        a_.set_title(NAMES[j].capitalize())
        a_.legend(fontsize=7, loc="upper center")
        a_.grid(alpha=0.3)
        tw = a_.twinx()
        for label, d in runs.items():
            tw.plot(t, np.rad2deg(d["U"].reshape(N, nu)[:, j]), "--",
                    color=cols[label], lw=1.0, alpha=0.55)
        tw.set_ylabel("Deflection [deg]", fontsize=8, color="0.35")
        tw.tick_params(axis="y", labelsize=7, colors="0.35")

    # Wall time, with the horizon-pass count printed on each bar so the timing
    # can be read against the work it corresponds to:
    bl = list(runs.keys())
    xb = np.arange(len(bl))
    gb = ax[5].bar(xb - 0.18, [runs[l]["t_grace"] for l in bl], 0.36,
                   color="crimson", label="GRACE")
    nb = ax[5].bar(xb + 0.18, [runs[l]["t_nlp"] for l in bl], 0.36,
                   color="steelblue", label="SLSQP")
    for bar, label in zip(gb, bl):
        ax[5].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{total_evals(runs[label]['c_grace'])}", ha="center",
                   va="bottom", fontsize=7)
    for bar, label in zip(nb, bl):
        tag = f"{total_evals(runs[label]['c_nlp'])}"
        if not runs[label]["nlp_ok"]:
            tag = tag + " failed"
            bar.set_hatch("//")
        ax[5].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), tag,
                   ha="center", va="bottom", fontsize=7)
    ax[5].set_yscale("log")
    ax[5].set_xticks(xb)
    ax[5].set_xticklabels(bl)
    ax[5].set_ylabel("Wall Time [s]")
    ax[5].set_title(f"Wall time at $N = {N}$ (horizon passes on bars)")
    ax[5].legend(fontsize=8)
    ax[5].grid(alpha=0.3, axis="y", which="both")

    fig.suptitle(f"Deviation around a keep-out, coupled cost saves "
                 f"{(1.0 - Cc.sum() / Cq.sum()) * 100:.1f}% under g",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figures/tests/lanechange_obstacle_cost.png", dpi=140,
                bbox_inches="tight")
    print("\nsaved figures/tests/lanechange_obstacle_cost.png")