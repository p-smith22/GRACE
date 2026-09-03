# Import packages:
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import grace

# === PLANT ===
# Lateral-directional model completed with the side-force equation, so the
# state carries sideslip and lateral position rather than attitude alone. The
# aileron produces no side force: it reaches y only through bank, while the
# rudder reaches y directly through sideslip. That asymmetry is the redundancy
# the coupled cost gets to spend.
IXX, IZZ = 4.0, 6.5
L_A, L_W, L_R = 0.90, 0.30, 0.05
N_A, N_W, N_R = -0.10, 0.55, 0.45
D_P, D_R = 1.6, 1.1
Y_B, Y_W, Y_R = -0.20, -0.05, 0.08
V_TAS, G_ACC = 45.0, 9.81

# Lateral offset is carried nondimensionally. Leaving y in meters against
# angles in radians makes the conditioning an artifact of unit choice:
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
        V_TAS * (psi + beta) / L_REF)

nx, nu = 6, 3
NAMES = ("aileron", "winglet", "rudder")
ZNAMES = (r"roll $\varphi$ [rad]", r"roll rate $p$ [rad/s]",
          r"heading $\psi$ [rad]", r"yaw rate $r$ [rad/s]",
          r"sideslip $\beta$ [rad]", r"offset $y/L$ [-]")

# The maneuver is a lane change: a lateral displacement flown and arrested,
# ending wings level, on heading, and coordinated. Bank-and-return against
# skid are two different ways to buy the same displacement, which is what
# gives the cost something to decide:
T_MAN = 8.0
N, dt = 40, T_MAN / 40
Y_OFF_M = 25.0
z0 = np.zeros(nx)
target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, Y_OFF_M / L_REF])
TIDX = [0, 1, 2, 3, 4, 5]

# === COST ===
# Hinge moment, the standard control-surface model. For surface i,
#
#     H_i = q S_i c_i ( Ch_alpha * alpha_i + Ch_delta * delta_i )
#
# with alpha_i the LOCAL incidence at that surface. The aileron and the
# winglet sit in each other's induced field, so deflecting one changes the
# incidence at the other through the downwash gradient,
#
#     alpha_a = alpha_0 - eps_w * delta_w,   alpha_w = alpha_0 - eps_a * delta_a
#
# Factoring q S c Ch_delta out of H leaves an effective deflection
# v = K u, which is the interference matrix used below with
#
#     alpha = eps_w * Ch_alpha / Ch_delta,   kappa * alpha = eps_a * Ch_alpha / Ch_delta
#
# so the coupling is a downwash gradient over a hinge-moment ratio rather
# than a chosen constant. Actuator work goes as H^2 for the aerodynamic part,
# plus a linear friction term that no aerodynamic model supplies:
RHO = 1.225
Q_DYN = 0.5 * RHO * V_TAS ** 2

# Surface geometry, area and mean chord aft of the hinge line:
S_SURF = np.array([0.32, 0.18, 0.22])
C_SURF = np.array([0.14, 0.11, 0.16])

# Hinge-moment derivatives, per radian. Both negative, which is the usual
# sign for an aerodynamically balanced surface:
CH_ALPHA = -0.28
CH_DELTA = -0.52

# Downwash gradients between the coupled pair, d(alpha_i)/d(delta_j). The
# winglet influences the aileron more strongly than the reverse, which is
# what makes the interference non-reciprocal:
EPS_W_ON_A = 1.67
EPS_A_ON_W = 0.75

# Interference strength and reciprocity ratio, both derived:
ALPHA = EPS_W_ON_A * CH_ALPHA / CH_DELTA
KAPPA = (EPS_A_ON_W / EPS_W_ON_A) ** 2

# Hinge-moment stiffness, the quadratic coefficient. This is the physical
# b_i: it scales with dynamic pressure, so the same cost reprices itself at
# a different flight condition with no refitting:
B_HNG = Q_DYN * S_SURF * C_SURF * abs(CH_DELTA)

# Breakout torque, in N m. Static friction and seal drag in the actuator, so
# unlike the terms above it is not aerodynamic and is taken from actuator
# data rather than derived:
A_BRK = np.array([1.85, 0.55, 1.05])

EPS_ABS = 0.02
THRESH = A_BRK / (2.0 * B_HNG)

def kmat(alpha=ALPHA):
    return np.array([[1.0,          -alpha, 0.0],
                     [-KAPPA * alpha, 1.0,  0.0],
                     [0.0,            0.0,  1.0]])

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

def step_costs(U, alpha=ALPHA):
    return phi_steps(np.asarray(U).reshape(-1, nu) @ kmat(alpha).T)

def coupled(alpha=ALPHA):
    K = kmat(alpha)
    KI = np.linalg.inv(K)

    def f(U):
        return float(np.sum(phi_steps(np.asarray(U).reshape(-1, nu) @ K.T)))

    def dinv(S):
        return (phi_inv(np.asarray(S).reshape(-1, nu) @ KI) @ KI.T).ravel()

    def blocks(S):
        V = phi_inv(np.asarray(S).reshape(-1, nu) @ KI)
        return np.einsum("ij,kj,lj->kil", KI, 1.0 / phi_hess(V), KI)

    return f, dinv, blocks

def grad_g(U, alpha=ALPHA):
    K = kmat(alpha)
    return (phi_grad(np.asarray(U).reshape(-1, nu) @ K.T) @ K).ravel()

# === CHECKS ===
def nlp_solve(system, tgt, f, grad, nsteps, maxiter=500):
    return minimize(f, np.zeros(nsteps * nu), jac=grad, method="SLSQP",
                    constraints=[dict(type="eq", fun=lambda v:
                        np.asarray(system.endpoint(v)) - system.target(tgt))],
                    options=dict(maxiter=maxiter, ftol=1e-10))

def stationarity(system, U, grad):
    _, Co = system.endpoint_jac(U)
    gr = grad(U)
    lam, *_ = np.linalg.lstsq(Co.T, gr, rcond=None)
    return (float(np.linalg.norm(gr - Co.T @ lam)
                  / max(np.linalg.norm(gr), 1e-30)), lam)

def endpoint_err(system, U, tgt):
    return float(np.linalg.norm(np.asarray(system.endpoint(U))
                                - system.target(tgt)))

# === RUN ===
if __name__ == "__main__":
    system = grace.build_cached(dynamics, nx=nx, nu=nu, N=N, z0=z0, dt=dt,
                                target_idx=TIDX, job="lanechange")
    engine = grace.GRACE(system)
    shoot = engine.shooting.lambda_shoot
    f_g, dinv_g, blk_g = coupled()

    print(f"lane change: {Y_OFF_M:.0f} m offset in {T_MAN:.0f} s at "
          f"{V_TAS:.0f} m/s, y normalized by L = {L_REF:.0f} m")
    print(f"K(alpha={ALPHA:.4f}) det = {np.linalg.det(kmat()):.4f}, "
          f"singular at alpha = {1.0 / np.sqrt(KAPPA):.3f}")
    print(f"hinge stiffness b = {np.array2string(B_HNG, precision=2)} N m/rad^2, "
          f"breakout threshold = "
          f"{np.array2string(np.rad2deg(THRESH), precision=2)} deg\n")

    runs = {}
    for label, kw, fn in (("quadratic", {}, lambda U: 0.5 * float(U @ U)),
                          ("coupled g", dict(cost=(f_g, dinv_g, blk_g)), f_g)):
        t0 = time.perf_counter()
        U = np.asarray(shoot(target, **kw)).flatten()
        tg = time.perf_counter() - t0
        lam = getattr(system, "_costate", None)
        if getattr(system, "_infeasible", False):
            print(f"WARNING: {label} reported infeasible")

        gr = (lambda V: V) if label == "quadratic" else grad_g
        t0 = time.perf_counter()
        res = nlp_solve(system, target, fn, gr, N)
        tn = time.perf_counter() - t0
        _, lam_n = stationarity(system, res.x, gr)

        runs[label] = dict(
            U=U, lam=None if lam is None else np.asarray(lam).ravel(),
            t_grace=tg, Z=np.asarray(system.rollout(U)),
            C=step_costs(U), C_nlp=step_costs(res.x),
            U_nlp=res.x, Z_nlp=np.asarray(system.rollout(res.x)),
            lam_nlp=lam_n, t_nlp=tn, its_nlp=res.nit,
            stat_own=stationarity(system, U, gr)[0],
            stat_g=stationarity(system, U, grad_g)[0],
            stat_nlp=stationarity(system, res.x, gr)[0],
            err=endpoint_err(system, U, target),
            dobj=abs(fn(U) - fn(res.x)) / max(abs(fn(res.x)), 1e-30),
            dU=float(np.abs(U - res.x).max()))

    cmin = min(float(d["C"].min()) for d in runs.values())
    print(f"minimum per-step channel cost across all runs: {cmin:.3e} "
          f"({'nonnegative' if cmin >= 0.0 else 'NEGATIVE -- BUG'})\n")

    print("both solutions priced under the coupled physical cost g")
    print(f"{'':<12}{'g(U)':>10}{'||U||^2':>10}{'shut':>7}"
          f"{'stat(own)':>11}{'stat(g)':>10}{'endpt':>10}")
    for label, d in runs.items():
        V = d["U"].reshape(-1, nu) @ kmat().T
        shut = np.mean(np.abs(V) < 0.25 * THRESH) * 100.0
        print(f"{label:<12}{f_g(d['U']):>10.5g}{float(d['U'] @ d['U']):>10.5g}"
              f"{shut:>6.0f}%{d['stat_own']:>11.1e}{d['stat_g']:>10.1e}"
              f"{d['err']:>10.1e}")

    # How the offset was bought. Bank-to-turn puts the displacement in through
    # heading and needs the roll undone at the end; skidding puts it in
    # through sideslip and leaves the wings closer to level:
    print(f"\nhow the offset was flown")
    print(f"{'':<12}{'peak |phi|':>12}{'peak |beta|':>13}"
          f"{'mean |beta|':>13}{'peak |psi|':>12}")
    for label, d in runs.items():
        Z = d["Z"]
        print(f"{label:<12}{np.abs(Z[:, 0]).max():>12.4f}"
              f"{np.abs(Z[:, 4]).max():>13.4f}"
              f"{np.abs(Z[:, 4]).mean():>13.4f}"
              f"{np.abs(Z[:, 2]).max():>12.4f}")

    Cq, Cc = runs["quadratic"]["C"], runs["coupled g"]["C"]
    print(f"\nper-surface cost under g")
    print(f"{'':<10}{'quadratic':>12}{'coupled g':>12}{'change':>10}")
    for j, nm in enumerate(NAMES):
        sq, sc = Cq[:, j].sum(), Cc[:, j].sum()
        print(f"{nm:<10}{sq:>12.4f}{sc:>12.4f}{(sc / sq - 1.0) * 100:>9.1f}%")
    print(f"{'total':<10}{Cq.sum():>12.4f}{Cc.sum():>12.4f}"
          f"{(Cc.sum() / Cq.sum() - 1.0) * 100:>9.1f}%")

    print("\nagreement with a cold-started SLSQP on the same cost")
    print(f"{'':<12}{'d obj':>10}{'max dU':>10}{'d nu':>10}"
          f"{'stat NLP':>10}{'GRACE s':>9}{'NLP s':>8}{'speedup':>9}")
    for label, d in runs.items():
        dl = (float(np.linalg.norm(d["lam"] - d["lam_nlp"])
                    / max(np.linalg.norm(d["lam_nlp"]), 1e-30))
              if d["lam"] is not None else np.nan)
        print(f"{label:<12}{d['dobj']:>10.1e}{d['dU']:>10.1e}{dl:>10.1e}"
              f"{d['stat_nlp']:>10.1e}{d['t_grace']:>9.3f}{d['t_nlp']:>8.3f}"
              f"{d['t_nlp'] / max(d['t_grace'], 1e-9):>8.1f}x")
    lg = runs["coupled g"]["lam"]
    if lg is not None:
        print(f"\ncostate, GRACE  {np.array2string(lg, precision=4)}")
        print(f"costate, NLP    "
              f"{np.array2string(runs['coupled g']['lam_nlp'], precision=4)}")

    # === PLOT: COST ===
    t = np.arange(N) * dt
    tz = np.arange(N + 1) * dt
    x_track = V_TAS * tz
    cols = {"quadratic": "0.45", "coupled g": "crimson"}

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.2))
    ax = ax.ravel()

    # Trajectory:
    for label, d in runs.items():
        ax[0].plot(x_track, L_REF * d["Z"][:, 5], "-", color=cols[label],
                   lw=2.2, label=f"{label}  (g = {f_g(d['U']):.3f})")
        ax[0].plot(x_track[::4], L_REF * d["Z_nlp"][::4, 5], "o",
                   color=cols[label], ms=3.5, mfc="none", lw=0)
    ax[0].axhline(Y_OFF_M, color="k", ls=":", lw=1.0)
    ax[0].plot(x_track[-1], Y_OFF_M, "*", color="k", ms=13, label="commanded")
    ax[0].set_xlabel("Y [m]")
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Trajectory")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # Total cost per step, both runs priced under g. This is the summary the
    # per surface panels break down, and the shaded area is the saving:
    for label, d in runs.items():
        s = d["C"].sum(axis=1)
        ax[1].plot(t, s, "-", color=cols[label], lw=2.2,
                   label=f"{label}  (total {s.sum():.2f})")
    ax[1].fill_between(t, Cc.sum(axis=1), Cq.sum(axis=1),
                       where=Cq.sum(axis=1) >= Cc.sum(axis=1),
                       color="crimson", alpha=0.12)
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("Control Cost [N m]")
    ax[1].set_title(f"Total Cost")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

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
            tw.plot(t, np.rad2deg(d["U"].reshape(-1, nu)[:, j]), "--",
                    color=cols[label], lw=1.0, alpha=0.55)
        tw.set_ylabel("Deflection [deg]", fontsize=8, color="0.35")
        tw.tick_params(axis="y", labelsize=7, colors="0.35")

    # Wall time at this horizon, from the solves already run:
    bl = list(runs.keys())
    xb = np.arange(len(bl))
    ax[5].bar(xb - 0.18, [runs[l]["t_grace"] for l in bl], 0.36,
              color="crimson", label="GRACE")
    ax[5].bar(xb + 0.18, [runs[l]["t_nlp"] for l in bl], 0.36,
              color="steelblue", label="Direct NLP")
    ax[5].set_yscale("log")
    ax[5].set_xticks(xb)
    ax[5].set_xticklabels(bl)
    ax[5].set_ylabel("Wall Time [s]")
    ax[5].set_title(f"Wall time at $N = {N}$")
    ax[5].legend(fontsize=8)
    ax[5].grid(alpha=0.3, axis="y", which="both")

    fig.suptitle(f"Lane change, {Y_OFF_M:.0f} m in {T_MAN:.0f} s.  "
                 f"Coupled cost saves "
                 f"{(1 - Cc.sum() / Cq.sum()) * 100:.1f}% under g",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figures/tests/lanechange_cost.png", dpi=140,
                bbox_inches="tight")

    # === PLOT: TRAJECTORY ===
    fig2, bx = plt.subplots(2, 3, figsize=(16.5, 7.2))
    bx = bx.ravel()
    for j in range(nx):
        for label, d in runs.items():
            bx[j].plot(tz, d["Z"][:, j], "-", color=cols[label], lw=2.2,
                       label=f"{label}, GRACE")
            bx[j].plot(tz[::4], d["Z_nlp"][::4, j], "o", color=cols[label],
                       ms=3.5, mfc="none", lw=0, label=f"{label}, NLP")
        bx[j].plot(tz[-1], target[j], "*", color="k", ms=13, label="target")
        bx[j].axhline(0.0, color="k", lw=0.8)
        bx[j].set_xlabel("time [s]")
        bx[j].set_ylabel(ZNAMES[j])
        bx[j].set_title(ZNAMES[j].split()[0].strip("$\\"))
        bx[j].grid(alpha=0.3)
    bx[0].legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig("figures/tests/lanechange_traj.png", dpi=140,
                 bbox_inches="tight")

    print("\nsaved figures/tests/lanechange_cost.png, lanechange_traj.png")

    # === REACHABILITY ===
    fig3, cx = plt.subplots(1, 2, figsize=(11.5, 4.8))
    reached = {}

    for label, d in runs.items():

        c = None if label == "quadratic" else (f_g, dinv_g, blk_g)
        E = 0.5 * float(d["U"] @ d["U"]) if c is None else f_g(d["U"])
        engine.reachability.print_summary(d["U"], name=label, cost=c,
                                          lam=d["lam"])
        r = engine.reachability.reach(d["U"], E, cost=c, lam=d["lam"],
                                      dims=(2, 5))
        reached[label] = (r, E)
        print(f"  calibration, endpoint form: {r['calibration']:.4f}")
        print(f"  calibration, quadrature   : "
              f"{r['calibration_quadrature']:.4f}")
        print(f"  identity residual         : {r['identity_error']:.2e} "
              f"(0 if the ray identity holds)")
        print(f"  calibration spread        : "
              f"{r['calibration_spread'][0]:.3f} to "
              f"{r['calibration_spread'][1]:.3f}")
        print(f"  ellipsoid departure       : max "
              f"{r['gap_ellipse'][0] * 100:.1f}%, "
              f"mean {r['gap_ellipse'][1] * 100:.1f}%")
        print(f"  calibrated departure      : max "
              f"{r['gap_calibrated'][0] * 100:.1f}%, "
              f"mean {r['gap_calibrated'][1] * 100:.1f}%")
        print(f"  off-plane residual        : {r['residual']:.1e}")
        print(f"  central symmetry          : {r['symmetry']:.2e}")
        print(f"  convexity, min turn ratio : {r['convexity']:+.2e}\n")

    # Left panel, the two true sets:
    for label, (r, E) in reached.items():
        cx[0].plot(np.rad2deg(r["true"][:, 0]), L_REF * r["true"][:, 1], "-",
                   color=cols[label], lw=2.2,
                   label=f"{label}  (budget {E:.3g})")
    cx[0].plot(np.rad2deg(target[0]), Y_OFF_M, "*", color="k", ms=13,
               label="achieved endpoint")
    cx[0].axhline(0.0, color="k", lw=0.7)
    cx[0].axvline(0.0, color="k", lw=0.7)
    cx[0].set_xlabel("heading [deg]")
    cx[0].set_ylabel("lateral offset [m]")
    cx[0].set_title("True reachable set under each cost,\n"
                    "each at the budget its own run spent")
    cx[0].legend(fontsize=8)
    cx[0].grid(alpha=0.3)

    # Right panel, the coupled set against its Gramian model:
    r_c, E_c = reached["coupled g"]
    cx[1].plot(np.rad2deg(r_c["ellipse"][:, 0]),
               L_REF * r_c["ellipse"][:, 1], "--", color="0.55", lw=1.6,
               label="Gramian ellipsoid")
    cx[1].plot(np.rad2deg(r_c["calibrated"][:, 0]),
               L_REF * r_c["calibrated"][:, 1], "-.", color="steelblue",
               lw=1.6, label=rf"anchored ($c={r_c['calibration']:.2f}$)")
    cx[1].plot(np.rad2deg(r_c["true"][:, 0]), L_REF * r_c["true"][:, 1], "-",
               color=cols["coupled g"], lw=2.2, label="true reachable set")
    cx[1].plot(np.rad2deg(target[0]), Y_OFF_M, "*", color="k", ms=13,
               label="achieved endpoint")
    cx[1].axhline(0.0, color="k", lw=0.7)
    cx[1].axvline(0.0, color="k", lw=0.7)
    cx[1].set_xlabel("heading [deg]")
    cx[1].set_ylabel("lateral offset [m]")
    cx[1].set_title(f"Coupled cost, budget {E_c:.3g}: mean radial error "
                    f"{r_c['gap_ellipse'][1] * 100:.1f}% "
                    f"to {r_c['gap_calibrated'][1] * 100:.1f}%")
    cx[1].legend(fontsize=8)
    cx[1].grid(alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("figures/tests/lanechange_reach.png", dpi=140,
                 bbox_inches="tight")
    print("saved figures/tests/lanechange_reach.png")