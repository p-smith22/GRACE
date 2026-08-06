# Import packages:
import numpy as np
import casadi as ca
import os
import grace
import time

# Shared implementation -- power-limited drive force, quadratic drag, bicycle steering:
def car(x, u):
    m = 1500.0
    P = 8000.0
    cd = 0.4
    L = 2.5
    delta_max = 0.6

    v = x[3]

    # Steering saturates at the rack limit, so the tan pole at pi/2 is
    # unreachable.  Fencing it with a bound instead does not work: the solve can
    # land past the pole and nothing can pull it back through infinity.
    delta = delta_max * ca.tanh(u[1] / delta_max)

    F_drive = P * u[0] / (ca.fabs(v) + 1.0)
    F_drag = cd * v * ca.fabs(v)

    return ca.vertcat(
        v * ca.cos(x[2]),
        v * ca.sin(x[2]),
        v / L * ca.tan(delta),
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
    system = grace.build_cached(car, nx=4, nu=2, N=N, z0=[0, 0, 0, 5.0], dt=0.1, job=job_name)
    engine = grace.GRACE(system)

    # === SIMPLE TRAJECTORY ===
    # Define target:
    target = np.array([30.0, 8.0, 0., 6.0])

    # Find optimal control:
    start = time.time()
    U = engine.shooting.lambda_shoot(target)
    Z = system.rollout(U)
    print(f"SIMPLE CASE ({time.time() - start:.2f}s): {engine.utils.diagnostics(U, target)}")

    # Plot:
    engine.utils.plotting(
        Z, U, pos_idx=(0, 1),
        state_names=["x", "y", "theta", "v"], control_names=["throttle", "steer"],
        title="Car -- minimum-effort drive",
        save=f"figures/{job_name}/simple.png",
    )

    # === OBSTACLE AVOIDANCE ===
    # Define target:
    target = np.array([30.0, 0.0, 0., 6.0])

    # Define obstacle:
    obstacles = [[16.0, 0.01]]
    R_obs = 6.0

    # Keep-out zone and actuator limits:
    constraints = [
                      (lambda z, u, o=o: R_obs ** 2 - ((z[0] - o[0]) ** 2 + (z[1] - o[1]) ** 2))
                      for o in obstacles
                  ] + [
                      lambda z, u: u[0] - 1.0,
                      lambda z, u: 0.0 - u[0],
                  ]

    # Find optimal trajectory to avoid obstacle:
    U_obs = engine.shooting.lambda_shoot(target, constraints=constraints, outer=60, inner=30)
    Z_obs = system.rollout(U_obs)
    print(f"SIMPLE CASE ({time.time() - start:.2f}s): {engine.utils.diagnostics(U_obs, target, constraints)}")

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