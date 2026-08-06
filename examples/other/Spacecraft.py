# Import packages:
import os
import numpy as np
import casadi as ca
import grace
import time

# Free-flying spacecraft:
def free_flyer(x, u):
    return ca.vertcat(x[2], x[3], u[0], u[1])

# Main run function:
def main():

    # Define job name:
    job_name = "spacecraft"

    # Define figures directory:
    os.makedirs(f"figures/{job_name}", exist_ok=True)

    # Define length of trajectory:
    N, dt = 150, 0.2

    # Build system and GRACE engine, cached:
    system = grace.build_cached(
        free_flyer, nx=4, nu=2, N=N, z0=[0, 0, 0, 0], dt=dt,
        job=job_name,
    )
    engine = grace.GRACE(system)

    # Define target and obstacles:
    target = np.array([45.0, 0.0, 0.0, 0.0])
    obstacle = [[15.0, 1.0], [30.0, -1.0]]
    R = 4.0

    # Keep-out zones as constraints.  Each is an expression g(z, u) <= 0, with
    # the centre and radius closed over so every lambda keeps its own:
    constraints = [
        (lambda z, u, o=o: R ** 2 - ((z[0] - o[0]) ** 2 + (z[1] - o[1]) ** 2))
        for o in obstacle
    ]

    # Lambda shoot for obstacle avoidance:
    start = time.time()
    U = engine.shooting.lambda_shoot(target, constraints=constraints, outer=50, inner=20)
    Z = system.rollout(U)
    print(f"OBSTACLE AVOIDANCE CASE ({time.time() - start:.2f}s): {engine.utils.diagnostics(U, target, constraints)}")

    # Plot:
    engine.utils.plotting(
        Z, U, obstacles=obstacle, R=R, pos_idx=(0, 1),
        state_names=["x", "y", "vx", "vy"], control_names=["thrust_x", "thrust_y"],
        title="Free-flying spacecraft",
        save=f"figures/{job_name}/obstacle_avoidance.png",
    )

# Run on call:
if __name__ == "__main__":
    main()
