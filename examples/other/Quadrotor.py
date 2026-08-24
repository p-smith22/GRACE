# Import packages:
import numpy as np
import casadi as ca
import os
import grace
import time

# Planar quadrotor dynamics:
def quadrotor(x, u):
    return ca.vertcat(x[2], x[3], -u[0] * ca.sin(x[4]), u[0] * ca.cos(x[4]) - 9.81, x[5], u[1])

# Define quadrotor dynamics with parameter, p, for thrust gain:
def quadrotor_p(x, u, p):
    return ca.vertcat(x[2], x[3], -p * u[0] * ca.sin(x[4]), p * u[0] * ca.cos(x[4]) - 9.81, x[5], u[1])

# Main run function:
def main():

    # Define job name:
    job_name = "quadrotor"

    # Define plotting directory:
    os.makedirs(f"figures/{job_name}", exist_ok=True)

    # Define length of trajectory:
    N, dt = 120, 0.05

    # Build system and engine (and cache):
    system = grace.build_cached(
        quadrotor, nx=6, nu=2, N=N, z0=[0, 0, 0, 0, 0, 0], dt=dt,
        job=job_name
    )
    engine = grace.GRACE(system)

    # Define target:
    target = np.array([4.0, 5.0, 0.0, 0.0, 0.0, 0.0])

    # === OPTIMAL CONTROL TO SIMPLE TARGET ===
    # Solve for optimal control sequence to target:
    start = time.time()
    U = engine.shooting.lambda_shoot(target)
    print(f"SIMPLE CASE ({time.time() - start:.2f}s): {engine.utils.diagnostics(U, target)}")

    # Plot:
    Z = system.rollout(U)
    engine.utils.plotting(
        Z, U, pos_idx=(0, 1),
        state_names=["x", "y", "vx", "vy", "phi", "omega"],
        control_names=["thrust", "torque"],
        title="Planar quadrotor -- minimum-effort maneuver",
        save=f"figures/{job_name}/simple.png",
    )

    # === LQR TRACKING DEMO ===
    # Define weights:
    Q = np.diag([200, 200, 20, 20, 5, 5])
    R = 0.01 * np.eye(2)
    Qf = 1000000 * Q

    # Fetch gains:
    gains, Z_nom = engine.tracking.lqr_gains(U, Q, R, Qf)

    # Plot with disturbance:
    rng = np.random.default_rng(0)
    disturb = rng.standard_normal((N, 6)) * 0.02
    Z_ol, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=False)
    Z_cl, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=True)
    engine.utils.closed_loop_plot(
        np.array(Z_nom), Z_ol, Z_cl, comp_idx=0, comp_name="x position",
        save=f"figures/{job_name}/tracking.png",
    )

    # === CODESIGN ===
    # Perform codesign on thrust gain (but penalize its use for conflicting objectives):
    codesign = grace.Codesign(quadrotor_p, nx=6, nu=2, N=N, z0=[0, 0, 0, 0, 0, 0], dt=0.05)
    U_cd, p_opt, front, sweep = codesign.optimize(
        target=target, param_name="p", objective=lambda p: 100 + 2000.0 * (p - 1.0) ** 2,
        p0=1.25, p_bounds=(0.5, 2.0), weights=np.linspace(0, 3, 50),
        save=f'figures/{job_name}/codesign.png', debug=True)

    # Print optimal thrust gain:
    gains_swept = [round(fp["param"], 2) for fp in front]
    print(f"codesign       : optimal thrust gain p = {p_opt:.3f}, front sweeps {gains_swept}")

    # === OBSTACLE AVOIDANCE ===
    # Define obstacle:
    obstacles = [[4.0, 0.5], [8.0, -0.5], [12.0, 0.5]]
    R_obs = 1
    target = np.array([16.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Keep-out zones as constraints.  Each is an expression g(z, u) <= 0, with
    # the centre closed over so every lambda keeps its own:
    constraints = [
        (lambda z, u, o=o: R_obs ** 2 - ((z[0] - o[0]) ** 2 + (z[1] - o[1]) ** 2))
        for o in obstacles
    ]

    # Find optimal control and trajectory:
    start = time.time()
    U_obs = engine.shooting.lambda_shoot(target, constraints=constraints)
    Z_obs = system.rollout(U_obs)
    print(f"OBSTACLE AVOIDANCE CASE ({time.time() - start:.2f}s): {engine.utils.diagnostics(U_obs, target, constraints)}")

    # Plot:
    engine.utils.plotting(
        Z_obs, U_obs, obstacles=obstacles, R=R_obs, pos_idx=(0, 1),
        state_names=["x", "y", "vx", "vy", "phi", "omega"],
        control_names=["thrust", "torque"],
        title="Planar quadrotor -- obstacle avoidance",
        save=f"figures/{job_name}/obstacle_avoidance.png",
    )

    # === REACHABILITY ANALYSIS ===
    engine.reachability.print_summary(U, name="quadrotor")

# Run on call:
if __name__ == "__main__":
    main()