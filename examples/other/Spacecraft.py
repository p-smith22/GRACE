# Import packages:
import os
import numpy as np
import casadi as ca
import grace

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
    N, dt = 80, 0.2

    # Build system and GRACE engine, cached:
    system = grace.build_cached(
        free_flyer, nx=4, nu=2, N=N, z0=[0, 0, 0, 0], dt=dt,
        job=job_name, pos_idx=(0, 1),
    )
    engine = grace.GRACE(system)

    # Define target and obstacles:
    target = np.array([10.0, 0.0, 0.0, 0.0])
    obstacle = [[5.0, 0.0]]
    R = 1.5

    # Lambda shoot for obstacle avoidance:
    U = engine.shooting.lambda_shoot(target, obstacles=obstacle, R=R,
                                     pos_idx=(0, 1))
    Z = system.rollout(U)

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