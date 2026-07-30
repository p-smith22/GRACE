# Import packages:
import numpy as np
import casadi as ca
import os
import grace

# Shared implementation -- power-limited drive force, quadratic drag, bicycle steering:
def car(x, u):

    # Define parameters:
    m = 1500.0  # weight
    P = 8000.0  # W (engine power)
    cd = 0.4  # aerodynamic drag coefficient
    L = 2.5  # wheelbase

    # Unpack the forward speed:
    v = x[3]

    # Drive force from a power-limited engine:
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

# Main run function:
def main():

    # Define job name:
    job_name = "car"

    # Make directory:
    os.makedirs(f"figures/{job_name}", exist_ok=True)

    # Define trajectory:
    N, dt = 60, 0.1

    # Build system and engine:
    system = grace.build_cached(car, nx=4, nu=2, N=N, z0=[0, 0, 0, 5.0], dt=0.1, pos_idx=(0, 1), job=job_name)
    engine = grace.GRACE(system)

    # === SIMPLE TRAJECTORY ===
    # Define target:
    target = np.array([30.0, 0.0, 0., 6.0])

    # Find optimal control:
    U = engine.shooting.newton_shoot(target)
    Z = system.rollout(U)

    # Plot:
    engine.utils.plotting(
        Z, U, pos_idx=(0, 1),
        state_names=["x", "y", "theta", "v"], control_names=["throttle", "steer"],
        title="Car -- minimum-effort drive",
        save=f"figures/{job_name}/simple.png",
    )

    # === OBSTACLE AVOIDANCE ===
    # Define obstacle:
    obstacles = [[16.0, 0.0]]
    R_obs = 4.5

    # Find optimal trajectory to avoid obstacle:
    U_obs = engine.shooting.lambda_shoot(target, obstacles=obstacles, R=R_obs,
                                         u_lo=[0.0, -0.6], u_hi=[1.0, 0.6])
    Z_obs = system.rollout(U_obs)

    # Plot:
    engine.utils.plotting(
        Z_obs, U_obs, obstacles=obstacles, R=R_obs, pos_idx=(0, 1),
        state_names=["x", "y", "theta", "v"], control_names=["throttle", "steer"],
        title="Car -- obstacle avoidance",
        save=f"figures/{job_name}/obstacle_avoidance.png",
    )

# Run on call:
if __name__ == "__main__":
    main()