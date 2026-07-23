# ============================================================================
# Spacecraft.py -- orbital rendezvous with a keep-out zone
# ============================================================================
# Two spacecraft problems through the same GRACE interface:
#   * a free-flying vehicle translating to a target pose while avoiding a keep-out
#     sphere (thrusters on both axes),
#   * a Clohessy-Wiltshire rendezvous, the standard linearized relative-orbital-motion
#     model, approaching a target while holding a keep-out radius around it.
# Both are minimum-effort, both are solved first try with no per-system tuning.
# ============================================================================

import os

import numpy as np
import casadi as ca

import grace


# Free-flying spacecraft: double integrator, thrusters along both body axes:
def free_flyer(x, u):
    return ca.vertcat(x[2], x[3], u[0], u[1])


# Clohessy-Wiltshire relative orbital motion about a circular reference orbit,
# with n the mean motion of the target orbit:
def clohessy_wiltshire(x, u):
    n = 0.0011                                   # rad/s, ~LEO mean motion
    return ca.vertcat(x[2], x[3],
                      3 * n ** 2 * x[0] + 2 * n * x[3] + u[0],
                      -2 * n * x[2] + u[1])


def main():
    os.makedirs("figures", exist_ok=True)

    # --- 1. Free flyer: translate 10 m while clearing a 1.5 m keep-out zone ---
    N = 80
    sys_ff = grace.build(free_flyer, nx=4, nu=2, N=N, z0=[0, 0, 0, 0], dt=0.2,
                         pos_idx=(0, 1), job="spacecraft_free_flyer")
    eng_ff = grace.GRACE(sys_ff)
    target_ff = np.array([10.0, 0.0, 0.0, 0.0])   # arrive at rest, 10 m downrange
    keepout_ff = [[5.0, 0.0]]
    R_ff = 1.5
    U_ff = eng_ff.shooting.lambda_shoot(target_ff, obstacles=keepout_ff, R=R_ff,
                                        pos_idx=(0, 1))
    Z_ff = sys_ff.rollout(U_ff)
    clr_ff = min((np.sum((Z_ff[:, :2] - np.asarray(o)) ** 2, axis=1) ** 0.5).min()
                 for o in keepout_ff)
    print(f"free flyer     : clearance {clr_ff:.3f} (need >= {R_ff}), "
          f"endpoint error {np.linalg.norm(sys_ff.endpoint(U_ff) - target_ff):.2e}, "
          f"effort {float(U_ff @ U_ff):.4f}")

    # --- 2. Clohessy-Wiltshire rendezvous with a 15 m keep-out sphere ---
    sys_cw = grace.build(clohessy_wiltshire, nx=4, nu=2, N=N, z0=[0, 0, 0, 0], dt=5.0,
                         pos_idx=(0, 1), job="spacecraft_rendezvous")
    eng_cw = grace.GRACE(sys_cw)
    target_cw = np.array([100.0, 0.0, 0.0, 0.0])  # 100 m along-track, arrive at rest
    keepout_cw = [[50.0, 0.0]]
    R_cw = 15.0
    U_cw = eng_cw.shooting.lambda_shoot(target_cw, obstacles=keepout_cw, R=R_cw,
                                        pos_idx=(0, 1))
    Z_cw = sys_cw.rollout(U_cw)
    clr_cw = min((np.sum((Z_cw[:, :2] - np.asarray(o)) ** 2, axis=1) ** 0.5).min()
                 for o in keepout_cw)
    print(f"cw rendezvous  : clearance {clr_cw:.3f} (need >= {R_cw}), "
          f"endpoint error {np.linalg.norm(sys_cw.endpoint(U_cw) - target_cw):.2e}, "
          f"effort {float(U_cw @ U_cw):.6f}")

    # --- Plots ---
    eng_ff.utils.plotting(
        Z_ff, U_ff, obstacles=keepout_ff, R=R_ff, pos_idx=(0, 1),
        state_names=["x", "y", "vx", "vy"], control_names=["thrust_x", "thrust_y"],
        title="Free-flying spacecraft -- keep-out zone avoidance",
        save="figures/example_spacecraft_free_flyer.png",
    )
    eng_cw.utils.plotting(
        Z_cw, U_cw, obstacles=keepout_cw, R=R_cw, pos_idx=(0, 1),
        state_names=["x", "y", "vx", "vy"], control_names=["thrust_x", "thrust_y"],
        title="Clohessy-Wiltshire rendezvous -- keep-out zone avoidance",
        save="figures/example_spacecraft_rendezvous.png",
    )
    print("plots          : figures/example_spacecraft_free_flyer.png, "
          "figures/example_spacecraft_rendezvous.png")


if __name__ == "__main__":
    main()
