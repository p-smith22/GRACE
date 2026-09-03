# Import packages:
import numpy as np
import os
import grace
from Aircraft_Dynamics import f as aircraft, f_codesign, X0
import time

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
        target_idx=lane_idx, job=job_name,
    )
    engine = grace.GRACE(system)

    # Set target as same state but with a Y offset (banked turn, straighten out):
    target = X0.copy()
    target[10] = 10.0

    # Control limits as constraints, one expression per side per surface:
    ctrl_bounds = np.array([20, 15, 10])
    surface_limits = []
    for i, b in enumerate(ctrl_bounds):
        surface_limits.append(lambda z, u, i=i, b=b: u[i] - b)
        surface_limits.append(lambda z, u, i=i, b=b: -b - u[i])

    # Find optimal control to target with realistic bounds:
    start = time.time()
    U = engine.shooting.lambda_shoot(target, constraints=surface_limits)
    Z = system.rollout(U)
    print(f"SIMPLE CASE ({time.time() - start:.2f}s): {engine.utils.diagnostics(U, target, surface_limits)}")

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
        target_idx=obs_lane_idx, job=job_name,
    )
    eng_obs = grace.GRACE(sys_obs)

    # Predict downrange position and set target state:
    downrange = sys_obs.rollout(np.zeros(N_obs * 3))[-1, 9]
    target_obs = X0.copy()
    target_obs[9] = downrange
    target_obs[10] = 0.0

    # Define obstacle:
    obstacles = [[downrange * 1/3, 1.0],[downrange * 2/3, -1.0]]
    R_obs = 6.0

    # Keep-out zone in the X-Y plane, alongside the same surface limits.  The
    # solver does not distinguish them: both are expressions g(z, u) <= 0:
    obstacle_zones = [
        (lambda z, u, o=o: R_obs ** 2 - ((z[9] - o[0]) ** 2 + (z[10] - o[1]) ** 2))
        for o in obstacles
    ]
    constraints = obstacle_zones + surface_limits

    import cProfile, pstats
    cProfile.runctx("eng_obs.shooting.lambda_shoot(target_obs, "
                    "constraints=obstacle_zones + surface_limits)",
                    globals(), locals(), "/tmp/prof")
    pstats.Stats("/tmp/prof").sort_stats("cumtime").print_stats(20)

    # Solve:
    start = time.time()
    sys_obs._count_ls = True
    U_obs = eng_obs.shooting.lambda_shoot(
        target_obs, constraints=obstacle_zones + surface_limits, outer=15, inner=20)
    sys_obs._count_ls = False
    Z_obs = sys_obs.rollout(U_obs)
    print(f"OBSTACLE AVOIDANCE CASE ({time.time() - start:.2f}s): {eng_obs.utils.diagnostics(U_obs, target_obs, constraints)}")

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