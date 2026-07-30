# Import packages:
import numpy as np
import os
import grace
from Aircraft_Dynamics import f as aircraft, f_codesign, X0

# Main run function:
def main():

    # Define job name:
    job_name = 'aircraft'

    # Define figure directory:
    os.makedirs(f"figures/{job_name}", exist_ok=True)

    # === SIMPLE TARGET ===
    # Define trajectory length:
    N, dt = 60, 0.05

    # Set constrained states as p, q, r, phi, theta, psi, Y:
    lane_idx = [3, 4, 5, 6, 7, 8, 10]

    # Bulid system and engine (cached):
    system = grace.build_cached(
        aircraft, nx=12, nu=3, N=N, z0=list(X0), dt=dt,
        target_idx=lane_idx, pos_idx=(9, 10), job=job_name,
    )
    engine = grace.GRACE(system)

    # Set target as same state but with a Y offset (banked turn, straighten out):
    target = X0.copy()
    target[10] = 10.0

    # Find optimal control to target with realistic bounds:
    ctrl_bounds = np.array([20, 15, 10])
    U = engine.shooting.lambda_shoot(target, u_lo=-ctrl_bounds, u_hi=ctrl_bounds)
    Z = system.rollout(U)

    # Plot:
    engine.utils.plotting(
        Z, U, pos_idx=(9, 10),
        state_names=["u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "X", "Y", "Z"],
        control_names=["aileron", "elevator", "rudder"],
        title="6DOF aircraft -- turn 20 deg then level the wings",
        save=f"figures/{job_name}/simple.png",
    )

    # === OBSTACLE AVOIDANCE ===
    # Define job name:
    job_name = "aircraft_obstacle"

    # Define figure directory:
    os.makedirs(f"figures/{job_name}", exist_ok=True)

    # Define system and engine:
    N_obs = 200
    obs_lane_idx = [3, 4, 5, 6, 7, 8, 9, 10]
    sys_obs = grace.build_cached(
        aircraft, nx=12, nu=3, N=N_obs, z0=list(X0), dt=0.05,
        target_idx=obs_lane_idx, pos_idx=(9, 10), job=job_name,
    )
    eng_obs = grace.GRACE(sys_obs)

    # Predict downrange position and set target state:
    downrange = sys_obs.rollout(np.zeros(N_obs * 3))[-1, 9]
    target_obs = X0.copy()
    target_obs[9] = downrange
    target_obs[10] = 0.0

    # Define obstacle:
    obstacles = [[downrange * 0.5, 0.0]]
    R_obs = 8.0

    # Solve:
    U_obs = eng_obs.shooting.lambda_shoot(target_obs, obstacles=obstacles, R=R_obs,
                                          pos_idx=(9, 10), u_lo=-ctrl_bounds, u_hi=ctrl_bounds)
    Z_obs = sys_obs.rollout(U_obs)

    # Plot:
    engine.utils.plotting(
        Z_obs, U_obs, obstacles=obstacles, R=R_obs, pos_idx=(9, 10),
        state_names=["u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "X", "Y", "Z"],
        control_names=["aileron", "elevator", "rudder"],
        title="6DOF aircraft -- obstacle avoidance",
        save=f"figures/{job_name}/obstacle_avoidance.png",
    )

# Run on call:
if __name__ == "__main__":
    main()