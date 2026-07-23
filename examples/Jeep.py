# ============================================================================
# example_jeep.py -- jeep car: full GRACE suite with a power-vs-cost codesign
# ============================================================================
# A ground vehicle whose engine power is a design variable.  Shows simple shoot,
# minimum-effort shoot, LQR tracking under disturbance, reachability, and a
# physically meaningful codesign: size the engine power against its cost for a
# demanding drive.
#
#   state:   [x, y, theta, v]     control: [throttle, steer]
# ============================================================================
import numpy as np
import os

import grace

import casadi as ca

# Default design parameters:
MASS = 1500.0  # kg
POWER = 8000.0  # W (engine power)
DRAG = 0.4  # aerodynamic drag coefficient
WHEELBASE = 2.5  # m


# Jeep dynamics with fixed default parameters (for plain control tasks):
def jeep(x, u):
    return _jeep(x, u, MASS, POWER, DRAG, WHEELBASE)


# Jeep dynamics exposing engine power as a design parameter (for codesign):
def jeep_power(x, u, P):
    return _jeep(x, u, MASS, P, DRAG, WHEELBASE)


# Jeep dynamics exposing mass as a design parameter (for codesign):
def jeep_mass(x, u, m):
    return _jeep(x, u, m, POWER, DRAG, WHEELBASE)


# Shared implementation -- power-limited drive force, quadratic drag, bicycle steering:
def _jeep(x, u, m, P, cd, L):
    # Unpack the forward speed:
    v = x[3]

    # Drive force from a power-limited engine: force = power / speed, scaled by throttle.
    # The +1 keeps the force finite at low speed (a stand-in for a torque-limited launch):
    F_drive = P * u[0] / (ca.fabs(v) + 1.0)

    # Aerodynamic drag opposes motion and grows with speed squared:
    F_drag = cd * v * ca.fabs(v)

    # Kinematic bicycle for position/heading, Newton's law for the speed:
    return ca.vertcat(
        v * ca.cos(x[2]),
        v * ca.sin(x[2]),
        v / L * ca.tan(u[1]),
        (F_drive - F_drag) / m,
    )

def main():
    N = 60
    system = grace.build(
        jeep, nx=4, nu=2, N=N, z0=[0, 0, 0, 5.0], dt=0.1, pos_idx=(0, 1), job="jeep",
    )
    engine = grace.GRACE(system)

    # Drive to (30, 8), heading slightly left, cruising at 6 m/s:
    target = np.array([30.0, 0.0, 0., 6.0])

    # --- 1. Newton feasibility shoot ---
    U_n = engine.shooting.newton_shoot(target)
    print(f"newton_shoot   : endpoint error {np.linalg.norm(system.endpoint(U_n) - system.target(target)):.2e}")

    # --- 2. Minimum-effort shoot, throttle bounded to [0, 1] and steer to +/- 0.6 rad ---
    U = engine.shooting.lambda_shoot(target, u_lo=[0.0, -0.6], u_hi=[1.0, 0.6])
    print(f"lambda_shoot   : endpoint error {np.linalg.norm(system.endpoint(U) - system.target(target)):.2e}, "
          f"cost {float(U @ U):.2f}")

    # --- 3. LQR tracking under random disturbance ---
    Q = np.diag([10.0, 10.0, 1.0, 1.0]); R = np.eye(2)
    gains, Z_nom = engine.tracking.lqr_gains(U, Q, R); Z_nom = np.array(Z_nom)
    rng = np.random.default_rng(0)
    disturb = rng.standard_normal((N, 4)) * 0.02
    Z_ol, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=False)
    Z_cl, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=True)
    err_ol = np.linalg.norm(Z_ol - Z_nom, axis=1).mean()
    err_cl = np.linalg.norm(Z_cl - Z_nom, axis=1).mean()
    print(f"lqr_gains      : mean tracking error  open-loop {err_ol:.3f}  closed-loop {err_cl:.3f}")

    # --- 4. Reachability summary ---
    engine.reachability.print_summary(U, name="jeep")

    # --- 5. Codesign: size the engine power against its cost for a demanding drive ---
    # More power means less throttle effort but a heavier, pricier engine.  The weight
    # sweep traces how much engine the maneuver justifies.
    codesign = grace.Codesign(jeep_power, nx=4, nu=2, N=N, z0=[0, 0, 0, 5.0], dt=0.1)
    demanding = np.array([40.0, 10.0, 0.2, 10.0])   # reach farther and faster
    U_cd, P_opt, front = codesign.optimize(
        target=demanding, param_name="P",
        objective=lambda P: (P / 1000.0) ** 2 * 3.0,     # engine cost grows with power
        p0=8000.0, p_bounds=(4000.0, 14000.0),
        weights=np.linspace(0, 3, 7), plot=True,
    )
    powers = [int(round(fp["param"])) for fp in front]
    print(f"codesign       : optimal engine power = {P_opt:.0f} W, front sweeps {powers} W")

    # --- 6. Obstacle avoidance: reach the target while routing around an obstacle ---
    obstacles = [[16.0, 0.0]]
    R_obs = 4.5
    U_obs = engine.shooting.lambda_shoot(target, obstacles=obstacles, R=R_obs,
                                         u_lo=[0.0, -0.6], u_hi=[1.0, 0.6])
    Z_obs = system.rollout(U_obs)
    clearance = min((np.sum((Z_obs[:, :2] - np.asarray(o)) ** 2, axis=1) ** 0.5).min()
                    for o in obstacles)
    print(f"obstacle_shoot : clearance {clearance:.2f} (need >= {R_obs}), "
          f"endpoint error {np.linalg.norm(system.endpoint(U_obs) - system.target(target)):.2e}")

    # --- Plots ---
    os.makedirs("figures", exist_ok=True)
    Z = system.rollout(U)
    engine.utils.plotting(
        Z, U, pos_idx=(0, 1),
        state_names=["x", "y", "theta", "v"], control_names=["throttle", "steer"],
        title="Jeep -- minimum-effort drive",
        save="figures/example_jeep.png",
    )
    engine.utils.plotting(
        Z_obs, U_obs, obstacles=obstacles, R=R_obs, pos_idx=(0, 1),
        state_names=["x", "y", "theta", "v"], control_names=["throttle", "steer"],
        title="Jeep -- obstacle avoidance",
        save="figures/example_jeep_obstacles.png",
    )
    engine.utils.closed_loop_plot(
        Z_nom, Z_ol, Z_cl, comp_idx=1, comp_name="y position",
        save="figures/example_jeep_tracking.png",
    )
    print("plots          : figures/example_jeep.png, figures/example_jeep_obstacles.png, "
          "figures/example_jeep_tracking.png")


if __name__ == "__main__":
    main()