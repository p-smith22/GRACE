# ============================================================================
# lqr_montecarlo.py -- LQR tracking robustness over random disturbances
# ============================================================================
# Plans one nominal trajectory for a planar quadrotor through a keep-out disc,
# then replays it many times under random process noise, open loop and closed
# loop on the same noise draws. Reports mean and spread of tracking error,
# terminal error, and obstacle clearance.
# ============================================================================

import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace

# === PROBLEM ===
# Planar quadrotor: z = [x, y, th, vx, vy, om], u = [thrust, torque]:
G = 9.81
J_INV = 1.0 / 0.02

# Thrust is commanded as a deviation from hover, so u = 0 is steady flight.
# With absolute thrust the minimum-effort solution drives T toward zero, where
# the attitude terms lose all sensitivity and the linearization goes singular.
def dynamics(z, u):
    T = G + u[0]
    return ca.vertcat(z[3], z[4], z[5],
                      -T * ca.sin(z[2]),
                      T * ca.cos(z[2]) - G,
                      J_INV * u[1])

# Keep-out disc sitting between the start and the goal:
OBS = np.array([2.5, 0])
R_OBS = 1.0
TARGET = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Constraint expression: positive inside the disc, so g(z, u) <= 0 clears it:
keep_out = lambda z, u: R_OBS ** 2 - ((z[0] - OBS[0]) ** 2 + (z[1] - OBS[1]) ** 2)

# === MONTE CARLO SETTINGS ===
N_TRIALS = 300
SIG_W = np.array([0.0, 0.0, 0.0, 0.03, 0.03, 0.02])

# === RUN ===
if __name__ == "__main__":

    # Build the system and solve for the nominal trajectory:
    system = grace.build_cached(dynamics, nx=6, nu=2, N=60, z0=[0, 0, 0, 0, 0, 0],
                         dt=0.05, job="lqr_mc")
    engine = grace.GRACE(system)
    start = time.time()
    U = engine.shooting.lambda_shoot(TARGET, constraints=[keep_out], outer=200, inner=100)
    print(f"OBSTACLE AVOIDANCE CASE ({time.time() - start:.2f}s): {engine.utils.diagnostics(U, TARGET, [keep_out])}")

    # Tracking gains about that nominal:
    Q = np.diag([10.0, 10.0, 1.0, 1.0, 1.0, 0.1])
    R = np.diag([0.1, 0.1])
    gains, Znom = engine.tracking.lqr_gains(U, Q, R)
    Znom = np.asarray(Znom)

    # Replay the same noise draws open loop and closed loop:
    rng = np.random.default_rng(0)
    Zcl = np.zeros((N_TRIALS, system.N + 1, system.nx))
    Zol = np.zeros_like(Zcl)
    for i in range(N_TRIALS):
        dist = rng.standard_normal((system.N, system.nx)) * SIG_W
        Zcl[i], _ = engine.utils.simulate(U, gains, Znom, disturb=dist,
                                          feedback=True)
        Zol[i], _ = engine.utils.simulate(U, gains, Znom, disturb=dist,
                                          feedback=False)

    # === STATISTICS ===
    def stats(Z):
        err = np.linalg.norm(Z - Znom[None, :, :], axis=2)
        term = np.linalg.norm(Z[:, -1, :2] - TARGET[:2], axis=1)
        clear = np.hypot(Z[:, :, 0] - OBS[0],
                         Z[:, :, 1] - OBS[1]).min(axis=1) - R_OBS
        return err, term, clear

    err_cl, term_cl, clr_cl = stats(Zcl)
    err_ol, term_ol, clr_ol = stats(Zol)

    print(f"\n{N_TRIALS} trials, same noise draws for both")
    print(f"{'':24}{'closed loop':>22}{'open loop':>22}")
    print(f"{'mean tracking error':24}"
          f"{err_cl.mean():>10.4f} ±{err_cl.std():<10.4f}"
          f"{err_ol.mean():>10.4f} ±{err_ol.std():<10.4f}")
    print(f"{'terminal error':24}"
          f"{term_cl.mean():>10.4f} ±{term_cl.std():<10.4f}"
          f"{term_ol.mean():>10.4f} ±{term_ol.std():<10.4f}")
    print(f"{'min clearance':24}"
          f"{clr_cl.mean():>+10.4f} ±{clr_cl.std():<10.4f}"
          f"{clr_ol.mean():>+10.4f} ±{clr_ol.std():<10.4f}")
    print(f"{'trials inside the disc':24}"
          f"{int((clr_cl < 0).sum()):>10d}{'':11}"
          f"{int((clr_ol < 0).sum()):>10d}")
    print(f"{'terminal p95':24}"
          f"{np.percentile(term_cl, 95):>10.4f}{'':11}"
          f"{np.percentile(term_ol, 95):>10.4f}")

    # === PLOTS ===
    mu = Zcl.mean(axis=0)
    sd = Zcl.std(axis=0)
    t = np.arange(system.N + 1) * system.dt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # Trajectory cloud against the nominal and the disc:
    th = np.linspace(0, 2 * np.pi, 200)
    ax[0].fill(OBS[0] + R_OBS * np.cos(th), OBS[1] + R_OBS * np.sin(th),
               color="0.85", zorder=0)
    for i in range(min(N_TRIALS, 120)):
        ax[0].plot(Zcl[i, :, 0], Zcl[i, :, 1], "-", color="steelblue",
                   lw=0.5, alpha=0.15, zorder=1)
    ax[0].plot(Znom[:, 0], Znom[:, 1], "-", color="crimson", lw=2,
               label="nominal", zorder=3)
    ax[0].plot(mu[:, 0], mu[:, 1], "--", color="k", lw=1.3,
               label="closed-loop mean", zorder=3)
    ax[0].plot(*TARGET[:2], "*", color="darkgreen", ms=14, zorder=4)
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].set_title("Closed-loop trajectories")
    ax[0].legend(fontsize=9)
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3)

    # Tracking error over time, closed loop against open loop:
    for e, lab, col in [(err_cl, "closed loop", "steelblue"),
                        (err_ol, "open loop", "darkorange")]:
        ax[1].plot(t, e.mean(axis=0), "-", color=col, lw=1.8, label=lab)
        ax[1].fill_between(t, e.mean(axis=0) - 2 * e.std(axis=0),
                           e.mean(axis=0) + 2 * e.std(axis=0),
                           color=col, alpha=0.20)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("‖z - z_nom‖")
    ax[1].set_title("Tracking error, mean ±2σ")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)

    # Clearance distribution, with the disc boundary marked:
    bins = np.linspace(min(clr_ol.min(), clr_cl.min()),
                       max(clr_ol.max(), clr_cl.max()), 40)
    ax[2].hist(clr_ol, bins=bins, color="darkorange", alpha=0.6,
               label="open loop")
    ax[2].hist(clr_cl, bins=bins, color="steelblue", alpha=0.8,
               label="closed loop")
    ax[2].axvline(0.0, color="crimson", ls="--", lw=1.5, label="disc boundary")
    ax[2].set_xlabel("minimum clearance [m]")
    ax[2].set_ylabel("trials")
    ax[2].set_title("Obstacle clearance")
    ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("figures/lqr_montecarlo.png", dpi=140, bbox_inches="tight")
    print("\nsaved figures/lqr_montecarlo.png")