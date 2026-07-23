# ============================================================================
# example_quadrotor.py -- planar quadrotor: full GRACE suite on one system
# ============================================================================
# Shows simple shoot, minimum-effort shoot, LQR tracking, reachability summary,
# and codesign on a single 6-state underactuated system, to demonstrate that the
# framework plugs into an arbitrary user system and delivers every capability.
#
#   state:  [x, y, vx, vy, phi, omega]   (position, velocity, tilt, tilt rate)
#   control:[thrust, torque]
# ============================================================================
import numpy as np
import casadi as ca
import os

import grace


# Planar quadrotor dynamics (thrust along the body axis, gravity down):
def quadrotor(x, u):
    g = 9.81
    return ca.vertcat(
        x[2],
        x[3],
        -u[0] * ca.sin(x[4]),
        u[0] * ca.cos(x[4]) - g,
        x[5],
        u[1],
    )


def main():
    # Build the system: 6 states, 2 controls, 1.5 s horizon:
    N = 60
    system = grace.build(
        quadrotor, nx=6, nu=2, N=N, z0=[0, 0, 0, 0, 0, 0], dt=0.05,
        pos_idx=(0, 1), job="quadrotor",
    )
    engine = grace.GRACE(system)

    # Target: translate to (4, 2), come to rest, level attitude:
    target = np.array([4.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    # --- 1. Newton feasibility shoot (reach the target) ---
    U_newton = engine.shooting.newton_shoot(target)
    err = np.linalg.norm(system.endpoint(U_newton) - system.target(target))
    print(f"newton_shoot   : endpoint error {err:.2e}")

    # --- 2. Minimum-effort (lambda) shoot ---
    U = engine.shooting.lambda_shoot(target)
    err = np.linalg.norm(system.endpoint(U) - system.target(target))
    print(f"lambda_shoot   : endpoint error {err:.2e}, control cost {float(U @ U):.2f}")

    # --- 3. LQR feedback about the nominal trajectory ---
    Q = np.diag([10, 10, 1, 1, 1, 1.0])
    R = np.eye(2)
    gains, Z_nom = engine.tracking.lqr_gains(U, Q, R)
    print(f"lqr_gains      : synthesized {len(gains)} feedback gains")

    # --- 4. Reachability summary (controllability Gramian structure) ---
    engine.reachability.print_summary(U, name="quadrotor")

    # --- 5. Codesign: optimize a thrust-effectiveness parameter ---
    def quadrotor_p(x, u, p):
        g = 9.81
        return ca.vertcat(
            x[2], x[3],
            -p * u[0] * ca.sin(x[4]), p * u[0] * ca.cos(x[4]) - g,
            x[5], u[1],
        )

    codesign = grace.Codesign(quadrotor_p, nx=6, nu=2, N=N, z0=[0, 0, 0, 0, 0, 0], dt=0.05)
    U_cd, p_opt, front = codesign.optimize(
        target=target, param_name="p", objective=lambda p: 2000.0 * (p - 1.0) ** 2,
        p0=1.0, p_bounds=(0.5, 2.0), weights=np.linspace(0, 3, 6), plot=False,
    )
    gains_swept = [round(fp["param"], 2) for fp in front]
    print(f"codesign       : optimal thrust gain p = {p_opt:.3f}, front sweeps {gains_swept}")

    # --- 6. Obstacle avoidance: reach the target while routing around an obstacle ---
    obstacles = [[2.0, 1.3]]
    R_obs = 0.7
    U_obs = engine.shooting.lambda_shoot(target, obstacles=obstacles, R=R_obs)
    Z_obs = system.rollout(U_obs)
    clearance = min((np.sum((Z_obs[:, :2] - np.asarray(o)) ** 2, axis=1) ** 0.5).min()
                    for o in obstacles)
    print(f"obstacle_shoot : clearance {clearance:.2f} (need >= {R_obs}), "
          f"endpoint error {np.linalg.norm(system.endpoint(U_obs) - system.target(target)):.2e}")

    # --- Plots: trajectory + states/controls, and closed-loop tracking ---
    os.makedirs("figures", exist_ok=True)
    Z = system.rollout(U)
    engine.utils.plotting(
        Z, U, pos_idx=(0, 1),
        state_names=["x", "y", "vx", "vy", "phi", "omega"],
        control_names=["thrust", "torque"],
        title="Planar quadrotor -- minimum-effort maneuver",
        save="figures/example_quadrotor.png",
    )
    engine.utils.plotting(
        Z_obs, U_obs, obstacles=obstacles, R=R_obs, pos_idx=(0, 1),
        state_names=["x", "y", "vx", "vy", "phi", "omega"],
        control_names=["thrust", "torque"],
        title="Planar quadrotor -- obstacle avoidance",
        save="figures/example_quadrotor_obstacles.png",
    )
    # closed-loop tracking under disturbance
    rng = np.random.default_rng(0)
    disturb = rng.standard_normal((N, 6)) * 0.02
    Z_ol, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=False)
    Z_cl, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=True)
    engine.utils.closed_loop_plot(
        np.array(Z_nom), Z_ol, Z_cl, comp_idx=0, comp_name="x position",
        save="figures/example_quadrotor_tracking.png",
    )
    print("plots          : figures/example_quadrotor.png, figures/example_quadrotor_tracking.png")


if __name__ == "__main__":
    main()