# Import packages:
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace
from grace.codesign.codesign import codesign

G = 9.81
MASS_BODY = 0.45

# Rotor spin-up time at a 0.10 m prop, and body drag per metre of arm:
TAU_0, CD_ARM = 0.08, 0.9

# === DESIGN VECTOR ===
# d = [L, T, R]
#   L : arm length [m]           more torque per unit thrust, more drag, more
#                                inertia because the motors sit further out
#   T : thrust capability        more acceleration authority, more motor mass
#   R : propeller radius [m]     more thrust per unit command, and a slower
#                                rotor, since spin-up time grows with radius
NAMES = ("arm length", "thrust cap", "prop radius")
D_LO = np.array([0.10, 0.8, 0.06])
D_HI = np.array([0.35, 2.4, 0.16])

# A poor starting airframe of the kind a design study exists to improve on:
# long arm, weak motor, small prop.
D_NOM = np.array([0.30, 0.95, 0.07])

# === PLANT ===
# z = [x, y, th, vx, vy, om, T_rotor], u = [thrust cmd, torque cmd].
#
# The design does more than scale the gains here. A bigger propeller makes more
# thrust and takes longer to spin up, and a longer arm is more frontal area, so
# two airframes with the same authority still fly differently: one has to
# anticipate and the other has to keep its speed down.
def dynamics(z, u, d):
    L, T, R = d[0], d[1], d[2]

    m_arm, m_motor, m_prop = 0.22 * L, 0.09 * T, 1.8 * R ** 2
    m_tot = MASS_BODY + 4.0 * (m_arm + m_motor + m_prop)
    Jz = 4.0 * (m_motor + m_prop) * L ** 2 + 0.02 * m_tot

    k_thrust = T * (R / 0.10) ** 2 / m_tot
    k_torque = T * L * (R / 0.10) ** 2 / Jz
    tau = TAU_0 * (R / 0.10) ** 2
    k_drag = CD_ARM * L

    thrust = G + z[6]
    return ca.vertcat(
        z[3], z[4], z[5],
        -thrust * ca.sin(z[2]) - k_drag * z[3] * ca.fabs(z[3]),
        thrust * ca.cos(z[2]) - G - k_drag * z[4] * ca.fabs(z[4]),
        k_torque * u[1],
        (k_thrust * u[0] - z[6]) / tau)

nx, nu, N, dt = 7, 2, 40, 0.05
z0 = np.zeros(nx)
target = np.array([4.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0])

# Rotor state left free at the end; it is internal, not part of the manoeuvre:
TIDX = list(range(6))

# === DESIGN OBJECTIVE ===
def objective(d):
    L, T, R = float(d[0]), float(d[1]), float(d[2])
    return MASS_BODY + 4.0 * (0.22 * L + 0.09 * T + 1.8 * R ** 2)

WEIGHT = 0.5

# === JOINT NLP BASELINE ===
# Controls and all three design parameters as decision variables at once.
def joint_nlp(nrm):
    U = ca.MX.sym("U", N * nu)
    d = ca.MX.sym("d", 3)

    z = ca.DM(z0)
    for k in range(N):
        uk = U[k * nu:(k + 1) * nu]
        k1 = dynamics(z, uk, d)
        k2 = dynamics(z + 0.5 * dt * k1, uk, d)
        k3 = dynamics(z + 0.5 * dt * k2, uk, d)
        k4 = dynamics(z + dt * k3, uk, d)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    m_tot = MASS_BODY + 4.0 * (0.22 * d[0] + 0.09 * d[1] + 1.8 * d[2] ** 2)
    w = ca.MX.sym("w", 1)

    # The same scalarization codesign solves, on the same normalization: a
    # Chebyshev objective in epigraph form. Comparing against a weighted sum of
    # raw ratios instead compares two different problems, and the two then pick
    # different points of the same front for reasons that have nothing to do
    # with either method's accuracy.
    Chat = (ca.dot(U, U) - nrm["C_id"]) / nrm["C_rng"]
    Dhat = (m_tot - nrm["D_id"]) / nrm["D_rng"]
    t = ca.MX.sym("t", 1)

    # The small extra term is what codesign carries as rho: without it the
    # objective is flat in whichever term is not the maximum, and the solver
    # can stop anywhere along that flat direction.
    obj = t + 1e-3 * (Chat + Dhat)
    g = ca.vertcat(z[TIDX] - ca.DM(target[TIDX]),
                   t - (1.0 - w) * Chat,
                   t - w * Dhat)

    S = ca.nlpsol("joint", "ipopt",
                  dict(x=ca.vertcat(U, d, t), p=w, f=obj, g=g),
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-8),
                       print_time=0))
    lbx = np.r_[np.full(N * nu, -np.inf), D_LO, 0.0]
    ubx = np.r_[np.full(N * nu, np.inf), D_HI, np.inf]
    lbg = np.r_[np.zeros(len(TIDX)), np.zeros(2)]
    ubg = np.r_[np.zeros(len(TIDX)), np.full(2, np.inf)]
    return S, lbx, ubx, lbg, ubg

# === RUN ===
if __name__ == "__main__":

    # Nominal airframe, for the normalization and the comparison:
    sys_nom = grace.build_cached(lambda z, u: dynamics(z, u, D_NOM),
                                 nx=nx, nu=nu, N=N, z0=z0, dt=dt,
                                 target_idx=TIDX, job="multi_nom")
    U_nom = np.asarray(
        grace.GRACE(sys_nom).shooting.lambda_shoot(target)).flatten()
    C_ref, D_ref = float(U_nom @ U_nom), objective(D_NOM)
    print("nominal design: " + ", ".join(
        f"{n} {v:.3f}" for n, v in zip(NAMES, D_NOM)))
    print(f"nominal effort {C_ref:.3f}, mass {D_ref:.3f} kg\n")

    # --- codesign, with the design as a vector ---
    # Swept over weights rather than run at one, because codesign and the NLP
    # below parameterize the trade differently: codesign scalarizes normalized
    # objectives, the NLP a weighted sum of raw ratios. A single weight puts
    # them at different points of the same front and the comparison says
    # nothing. Comparing the fronts themselves does not depend on either
    # parameterization.
    WSET = np.linspace(0.05, 0.95, 9)
    t0 = time.perf_counter()
    U_cd, d_cd, front, _ = codesign(
        dynamics, nx, nu, N, z0, dt, target, "airframe", objective,
        D_NOM, list(zip(D_LO, D_HI)), weights=WSET,
        norm="cheby", target_idx=TIDX, plot=False, job="multi")
    t_cd = time.perf_counter() - t0
    U_cd = np.asarray(U_cd).flatten()
    C_cd, D_cd = float(U_cd @ U_cd), objective(d_cd)
    F_cd = np.array([[f["cost"], f["objective"]] for f in front])

    # --- joint NLP, swept over its own weight ---
    t0 = time.perf_counter()
    S, lbx, ubx, lbg, ubg = joint_nlp(front[0]["norm"])
    t_build_jt = time.perf_counter() - t0

    t0 = time.perf_counter()
    xg = np.r_[np.zeros(N * nu), D_NOM, 1.0]
    F_jt, picks, n_fail = [], [], 0
    for w in WSET:
        sol = S(x0=xg, p=float(w), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        ok = S.stats()["success"]

        # A weight the NLP failed at is dropped rather than plotted. It
        # returns its last iterate either way, and that point can be worse than
        # the nominal on both objectives -- not a front point at all, and it
        # would otherwise be drawn as one. Its own answer is also reused as the
        # start for the next weight, so a failure left in place propagates.
        if not ok:
            n_fail += 1
            xg = np.r_[np.zeros(N * nu), D_NOM, 1.0]
            continue

        xg = np.array(sol["x"]).flatten()
        Uw, dw = xg[:N * nu], xg[N * nu:N * nu + 3]
        F_jt.append([float(Uw @ Uw), objective(dw)])
        picks.append((dw.copy(), Uw.copy()))
    t_jt = time.perf_counter() - t0
    F_jt = np.array(F_jt)
    if n_fail:
        print(f"[joint NLP] {n_fail} of {len(WSET)} weights did not converge "
              f"and were dropped")

    # Same dominance filter codesign applies to its own front, so neither is
    # credited with a point that is worse than another on both objectives:
    keep = [i for i in range(len(F_jt))
            if not np.any((F_jt[:, 0] <= F_jt[i, 0])
                          & (F_jt[:, 1] <= F_jt[i, 1])
                          & ((F_jt[:, 0] < F_jt[i, 0])
                             | (F_jt[:, 1] < F_jt[i, 1])))]
    n_dom = len(F_jt) - len(keep)
    if n_dom:
        print(f"[joint NLP] {n_dom} dominated point(s) dropped")
    F_jt = F_jt[keep]
    picks = [picks[i] for i in keep]

    # Point on the NLP front closest to the design codesign selected:
    k = int(np.argmin(np.linalg.norm(
        (picks_arr := np.array([p for p, _ in picks])) - d_cd, axis=1)))
    d_jt, U_jt = picks[k]
    C_jt, D_jt = F_jt[k, 0], F_jt[k, 1]

    def scal(C, D):
        return (1.0 - WEIGHT) * C / C_ref + WEIGHT * D / D_ref

    # === RESULTS ===
    print(f"{'':<14}{'codesign':>12}{'joint NLP':>12}{'nominal':>12}")
    for i, n in enumerate(NAMES):
        print(f"{n:<14}{d_cd[i]:>12.4f}{d_jt[i]:>12.4f}{D_NOM[i]:>12.4f}")
    print(f"{'effort':<14}{C_cd:>12.3f}{C_jt:>12.3f}{C_ref:>12.3f}")
    print(f"{'mass [kg]':<14}{D_cd:>12.3f}{D_jt:>12.3f}{D_ref:>12.3f}")
    print(f"{'scalarized':<14}{scal(C_cd, D_cd):>12.4f}"
          f"{scal(C_jt, D_jt):>12.4f}{scal(C_ref, D_ref):>12.4f}")
    print("  both solve the same Chebyshev scalarization on the same ideal "
          "and nadir,\n  so the columns are comparable weight for weight")

    # Build is a one time cost and is cached across runs, so it is reported
    # apart from the solve. Both methods trace the same nine point front, so
    # per front point is the number that compares them.
    n_w = len(WSET)
    print(f"\n{'':<24}{'build s':>10}{'solve s':>10}{'total s':>10}"
          f"{'ms per point':>14}")
    print(f"{'codesign (GRACE)':<24}{'cached':>10}{t_cd:>10.3f}"
          f"{t_cd:>10.3f}{t_cd / n_w * 1e3:>14.1f}")
    print(f"{'joint NLP (IPOPT)':<24}{t_build_jt:>10.3f}{t_jt:>10.3f}"
          f"{t_build_jt + t_jt:>10.3f}{t_jt / n_w * 1e3:>14.1f}")
    print(f"{'solve speedup':<24}{'':>10}"
          f"{1 / (t_cd / max(t_jt, 1e-12)):>10.2f}x")
    print(f"{'total speedup':<24}{'':>10}{'':>10}"
          f"{1 / (t_cd / max(t_build_jt + t_jt, 1e-12)):>10.2f}x")
    # Agreement measured as the effort each front reports at the same mass,
    # over the range both cover. Nearest neighbour in the plane is misleading
    # here: the two fronts run to different extremes, and a point past the end
    # of the other curve has no neighbour to be near, which reads as
    # disagreement when it is only a difference in extent.
    o_cd, o_jt = np.argsort(F_cd[:, 1]), np.argsort(F_jt[:, 1])
    m_lo = max(F_cd[:, 1].min(), F_jt[:, 1].min())
    m_hi = min(F_cd[:, 1].max(), F_jt[:, 1].max())
    mm = np.linspace(m_lo, m_hi, 40)
    e_cd = np.interp(mm, F_cd[o_cd, 1], F_cd[o_cd, 0])
    e_jt = np.interp(mm, F_jt[o_jt, 1], F_jt[o_jt, 0])
    rel = np.abs(e_cd - e_jt) / np.maximum(e_jt, 1e-12)
    print(f"\nfront agreement over the shared mass range "
          f"[{m_lo:.3f}, {m_hi:.3f}] kg: "
          f"mean {rel.mean() * 100:.2f}%, worst {rel.max() * 100:.2f}%")

    # Which of the two is actually on the front. At a given mass the lower
    # effort is the better design, so a signed comparison says whether the
    # difference is one method falling short or the other. Both are local
    # methods on a nonconvex problem, and the joint NLP takes a simultaneous
    # step in controls and design, which is not obviously the more reliable of
    # the two.
    sign = (e_cd - e_jt) / np.maximum(e_jt, 1e-12)
    n_cd = int(np.sum(sign < -1e-6))
    n_jt = int(np.sum(sign > 1e-6))
    print(f"  at matched mass: codesign lower effort at {n_cd} of {len(mm)} "
          f"samples, joint NLP lower at {n_jt}")
    print(f"  mean signed difference {sign.mean() * 100:+.2f}% "
          f"(negative means codesign is below the NLP front)")
    print(f"nearest NLP design to the codesign pick: "
          f"max |dd| = {np.abs(d_cd - d_jt).max():.4f}")
    print(f"effort reduced {C_ref / max(C_cd, 1e-12):.1f}x at "
          f"{(1 - D_cd / D_ref) * 100:.1f}% less mass")

    # === PLOT ===
    def roll(d, U):
        sy = grace.build_cached(lambda z, u, _d=np.asarray(d): dynamics(z, u, _d),
                                nx=nx, nu=nu, N=N, z0=z0, dt=dt,
                                target_idx=TIDX,
                                job="multi_r" + "_".join(f"{v:.3f}" for v in d))
        return np.asarray(sy.rollout(np.asarray(U).flatten()))

    trajs = (("nominal", D_NOM, U_nom, "0.6"),
             ("codesign", d_cd, U_cd, "steelblue"),
             ("joint NLP", d_jt, U_jt, "darkorange"))
    Z = {lab: roll(d, U) for lab, d, U, _ in trajs}
    tt = np.arange(N + 1) * dt

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
    ax = ax.ravel()

    # Raw values, one small axis per parameter. Normalizing all three onto one
    # axis to share a scale hides what the numbers are: an arm length of 0.30 m
    # plots as 0.80 of its range, which reads as a much larger design than it
    # is. The bounds are drawn so the position within the range is still
    # visible without having to encode it.
    ax[0].remove()
    gs = ax[1].get_gridspec()
    subs = gs[0, 0].subgridspec(1, 3, wspace=0.55)
    UNITS = ("m", "x hover", "m")
    for j, (nm, un) in enumerate(zip(NAMES, UNITS)):
        a = fig.add_subplot(subs[0, j])
        vals = [D_NOM[j], d_cd[j], d_jt[j]]
        a.bar(["nom", "cd", "NLP"], vals,
              color=["0.6", "steelblue", "darkorange"], width=0.65)
        a.axhline(D_LO[j], color="crimson", ls=":", lw=1.2)
        a.axhline(D_HI[j], color="crimson", ls=":", lw=1.2)
        for x_, v_ in enumerate(vals):
            a.text(x_, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=7)
        a.set_ylim(D_LO[j] - 0.12 * (D_HI[j] - D_LO[j]),
                   D_HI[j] + 0.18 * (D_HI[j] - D_LO[j]))
        a.set_title(f"{nm} [{un}]", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.3, axis="y")
        if j == 0:
            a.set_ylabel("Parameter Value", fontsize=7)

    labs = ["nominal", "codesign", "joint NLP"]
    cols3 = ["0.6", "steelblue", "darkorange"]
    ax[1].bar(labs, [C_ref, C_cd, C_jt], color=cols3)
    ax[1].set_ylabel("control effort")
    ax[1].set_title("Control effort")
    ax[1].grid(alpha=0.3, axis="y")

    ax[2].bar(labs, [D_ref, D_cd, D_jt], color=cols3)
    ax[2].set_ylabel("take-off mass [kg]")
    ax[2].set_title("Design Cost (Mass)")
    ax[2].grid(alpha=0.3, axis="y")

    for lab, _, _, col in trajs:
        ax[3].plot(Z[lab][:, 0], Z[lab][:, 1], "-", color=col, lw=2.0,
                   label=lab)
    ax[3].plot([target[0]], [target[1]], "*", color="crimson", ms=15,
               label="target")
    ax[3].set_xlabel("x [m]")
    ax[3].set_ylabel("y [m]")
    ax[3].set_title("Trajectory")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3)

    for lab, _, _, col in trajs:
        ax[4].plot(tt, np.rad2deg(Z[lab][:, 2]), "-", color=col, lw=2.0,
                   label=lab)
    ax[4].set_xlabel("time [s]")
    ax[4].set_ylabel("pitch [deg]")
    ax[4].set_title("Attitude")
    ax[4].legend(fontsize=8)
    ax[4].grid(alpha=0.3)

    ax[5].plot(F_cd[:, 1], F_cd[:, 0], "o-", color="steelblue", ms=5,
               label="codesign")
    ax[5].plot(F_jt[:, 1], F_jt[:, 0], "s--", color="darkorange", ms=5,
               label="joint NLP")
    ax[5].plot([D_ref], [C_ref], "*", color="0.4", ms=15, label="nominal")
    ax[5].set_xlabel("take-off mass [kg]")
    ax[5].set_ylabel("control effort")
    ax[5].set_yscale("log")
    ax[5].set_title("Pareto Front")
    ax[5].legend(fontsize=8)
    ax[5].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/tests/multi_codesign.png", dpi=140, bbox_inches="tight")
    print("\nsaved figures/tests/multi_codesign.png")