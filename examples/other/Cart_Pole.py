# Import packages:
import numpy as np
import casadi as ca
import os
import grace


# Cart-pole dynamics (pole hangs at theta = 0, upright at theta = pi):
def cartpole(x, u, l=0.5):
    g = 9.81; mc = 1.0; mp = 0.1
    s = ca.sin(x[1]); c = ca.cos(x[1]); den = mc + mp * s * s
    ddx = (u[0] + mp * s * (l * x[3] ** 2 + g * c)) / den
    ddth = (-u[0] * c - mp * l * x[3] ** 2 * c * s - (mc + mp) * g * s) / (l * den)
    return ca.vertcat(x[2], x[3], ddx, ddth)

# Main function:
def main():

    # Define job name:
    job_name = "cartpole"

    # Set directory:
    os.makedirs(f"figures/{job_name}", exist_ok=True)

    # Define trajectory (6 s covers the pumping a swing-up needs):
    N, dt = 30, 0.05

    # Define system and engine (build_cached actually writes the graph cache):
    system = grace.build_cached(cartpole, nx=4, nu=1, N=N, z0=[0, 0, 0, 0], dt=dt, job=job_name)
    engine = grace.GRACE(system)

    # Define target (pole starts hanging at 0 and finishes upright at pi):
    target = np.array([5.0, np.pi, 0.0, 0.0])

    # Generate control to target state:
    U = engine.shooting.lambda_shoot(target)
    Z = system.rollout(U)

    # Report the endpoint with theta wrapped, since -pi is upright too:
    err = Z[-1] - target
    err[1] = (err[1] + np.pi) % (2 * np.pi) - np.pi
    print(f"endpoint error {np.linalg.norm(err):.2e}, cost {float(U @ U):.2f}, peak force {np.abs(U).max():.2f} N")

    # Plot:
    engine.utils.plotting(
        Z, U, show_traj=False,
        state_names=["x", "theta", "x_dot", "theta_dot"], control_names=["force"],
        title="Cart-pole -- minimum-effort swing-up",
        save=f"figures/{job_name}/simple.png",
    )

    # Print reachability summary:
    engine.reachability.print_summary(U, name="cartpole")

# Run on call:
if __name__ == "__main__":
    main()
