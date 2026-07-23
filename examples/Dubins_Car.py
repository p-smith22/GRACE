# ============================================================================
# example_dubins.py -- Dubins car: full GRACE suite on a nonholonomic system
# ============================================================================
# A velocity-controlled ground robot -- shows the framework handles nonholonomic
# systems whose U=0 nominal is stationary.  Demonstrates simple shoot, min-effort
# shoot, LQR tracking under disturbance, reachability, and codesign.
#
#   state:  [x, y, theta]     control: [speed, yaw_rate]
# ============================================================================
import numpy as np
import casadi as ca
import os

import grace


# Dubins car dynamics (speed and yaw rate as controls):
def dubins(x, u):
    return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])


def main():
    N = 60
    system = grace.build(
        dubins, nx=3, nu=2, N=N, z0=[0, 0, 0], dt=0.05, pos_idx=(0, 1), job="dubins",
    )
    engine = grace.GRACE(system)

    # Target pose: reach (4, 3) heading pi/2:
    target = np.array([4.0, 3.0, np.pi / 2])

    # --- 1. Newton feasibility shoot ---
    U_n = engine.shooting.newton_shoot(target)
    print(f"newton_shoot   : endpoint error {np.linalg.norm(system.endpoint(U_n) - system.target(target)):.2e}")

    # --- 2. Minimum-effort shoot (optionally with a minimum-speed box bound) ---
    U = engine.shooting.lambda_shoot(target, u_lo=[0.0, -2.0], u_hi=[3.0, 2.0])
    print(f"lambda_shoot   : endpoint error {np.linalg.norm(system.endpoint(U) - system.target(target)):.2e}, "
          f"cost {float(U @ U):.2f}")

    # --- 3. LQR tracking under random disturbance ---
    Q = np.diag([10.0, 10.0, 1.0]); R = np.eye(2)
    gains, Z_nom = engine.tracking.lqr_gains(U, Q, R); Z_nom = np.array(Z_nom)
    rng = np.random.default_rng(0)
    disturb = rng.standard_normal((N, 3)) * 0.02
    Z_ol, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=False)
    Z_cl, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=True)
    err_ol = np.linalg.norm(Z_ol - Z_nom, axis=1).mean()
    err_cl = np.linalg.norm(Z_cl - Z_nom, axis=1).mean()
    print(f"lqr_gains      : mean tracking error  open-loop {err_ol:.3f}  closed-loop {err_cl:.3f}")

    # --- 4. Reachability summary ---
    engine.reachability.print_summary(U, name="dubins")

    # --- 5. Codesign: optimize a steering-effectiveness parameter ---
    def dubins_p(x, u, p):
        # p scales control effectiveness (both speed and yaw authority):
        return ca.vertcat(p * u[0] * ca.cos(x[2]), p * u[0] * ca.sin(x[2]), p * u[1])

    codesign = grace.Codesign(dubins_p, nx=3, nu=2, N=N, z0=[0, 0, 0], dt=0.05)
    U_cd, p_opt, front = codesign.optimize(
        target=target, param_name="p", objective=lambda p: 150.0 * (p - 1.0) ** 2,
        p0=1.0, p_bounds=(0.5, 2.0), weights=np.linspace(0, 3, 6), plot=True,
    )
    gains_swept = [round(fp["param"], 2) for fp in front]
    print(f"codesign       : actuator gain p = {p_opt:.3f}, front sweeps {gains_swept}")

    # --- 6. Obstacle avoidance: reach the target while routing around an obstacle ---
    # The detour needs more time than the open maneuver, so use a longer horizon:
    obstacles = [[2.0, 1.5]]
    R_obs = 0.9
    sys_obs = grace.build(
        dubins, nx=3, nu=2, N=100, z0=[0, 0, 0], dt=0.05, pos_idx=(0, 1), job="dubins_obstacle",
    )
    eng_obs = grace.GRACE(sys_obs)
    U_obs = eng_obs.shooting.lambda_shoot(target, obstacles=obstacles, R=R_obs,
                                          u_lo=[0.0, -2.0], u_hi=[3.0, 2.0])
    Z_obs = sys_obs.rollout(U_obs)
    clearance = min((np.sum((Z_obs[:, :2] - np.asarray(o)) ** 2, axis=1) ** 0.5).min()
                    for o in obstacles)
    print(f"obstacle_shoot : clearance {clearance:.2f} (need >= {R_obs}), "
          f"endpoint error {np.linalg.norm(sys_obs.endpoint(U_obs) - sys_obs.target(target)):.2e}")

    # --- Plots: trajectory + states/controls, and closed-loop tracking ---
    os.makedirs("figures", exist_ok=True)
    Z = system.rollout(U)
    engine.utils.plotting(
        Z, U, obstacles=None, pos_idx=(0, 1),
        state_names=["x", "y", "theta"], control_names=["speed", "yaw_rate"],
        title="Dubins car -- minimum-effort maneuver",
        save="figures/example_dubins.png",
    )
    engine.utils.plotting(
        Z_obs, U_obs, obstacles=obstacles, R=R_obs, pos_idx=(0, 1),
        state_names=["x", "y", "theta"], control_names=["speed", "yaw_rate"],
        title="Dubins car -- obstacle avoidance",
        save="figures/example_dubins_obstacles.png",
    )
    engine.utils.closed_loop_plot(
        Z_nom, Z_ol, Z_cl, comp_idx=0, comp_name="x position",
        save="figures/example_dubins_tracking.png",
    )
    print("plots          : figures/example_dubins.png, figures/example_dubins_obstacles.png, "
          "figures/example_dubins_tracking.png")


if __name__ == "__main__":
    main()