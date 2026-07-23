# ============================================================================
# example_cartpole.py -- cart-pole: full GRACE suite on an underactuated system
# ============================================================================
# A single control (cart force) drives four states -- shows how to target a
# subset of states (target_idx) on an underactuated system, plus min-effort
# shoot, LQR stabilization under disturbance, reachability, and codesign on the
# pole length.
#
#   state:  [x, theta, x_dot, theta_dot]     control: [force]
# ============================================================================
import numpy as np
import casadi as ca
import os

import grace


# Cart-pole dynamics (pole hangs at theta = pi, upright at theta = 0):
def cartpole(x, u, l=0.5):
    g = 9.81; mc = 1.0; mp = 0.1
    s = ca.sin(x[1]); c = ca.cos(x[1]); den = mc + mp * s * s
    ddx = (u[0] + mp * s * (l * x[3] ** 2 + g * c)) / den
    ddth = (-u[0] * c - mp * l * x[3] ** 2 * c * s - (mc + mp) * g * s) / (l * den)
    return ca.vertcat(x[2], x[3], ddx, ddth)


def main():
    N = 80
    # Underactuated: 1 control can't independently place all 4 endpoint states, so
    # target the pole angle (index 1) -- swing the pole upright:
    system = grace.build(
        lambda x, u: cartpole(x, u), nx=4, nu=1, N=N, z0=[0, np.pi, 0, 0], dt=0.05,
        target_idx=[1], job="cartpole",
    )
    engine = grace.GRACE(system)
    target = np.array([0.0])   # pole upright

    # --- 1. Newton feasibility shoot ---
    U_n = engine.shooting.newton_shoot(target)
    print(f"newton_shoot   : endpoint error {np.linalg.norm(system.endpoint(U_n) - system.target(target)):.2e}")

    # --- 2. Minimum-effort shoot ---
    U = engine.shooting.lambda_shoot(target)
    print(f"lambda_shoot   : endpoint error {np.linalg.norm(system.endpoint(U) - system.target(target)):.2e}, "
          f"cost {float(U @ U):.2f}")

    # --- 3. LQR stabilization under random disturbance ---
    Q = np.diag([1.0, 10.0, 1.0, 1.0]); R = np.eye(1)
    gains, Z_nom = engine.tracking.lqr_gains(U, Q, R); Z_nom = np.array(Z_nom)
    rng = np.random.default_rng(0)
    disturb = rng.standard_normal((N, 4)) * 0.01
    Z_ol, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=False)
    Z_cl, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=True)
    err_ol = np.linalg.norm(Z_ol - Z_nom, axis=1).mean()
    err_cl = np.linalg.norm(Z_cl - Z_nom, axis=1).mean()
    print(f"lqr_gains      : mean tracking error  open-loop {err_ol:.3f}  closed-loop {err_cl:.3f}")

    # --- 4. Reachability summary ---
    engine.reachability.print_summary(U, name="cartpole")

    # --- Plots: states/controls, and closed-loop stabilization ---
    os.makedirs("figures", exist_ok=True)
    Z = system.rollout(U)
    engine.utils.plotting(
        Z, U, show_traj=False,
        state_names=["x", "theta", "x_dot", "theta_dot"], control_names=["force"],
        title="Cart-pole -- minimum-effort swing-up",
        save="figures/example_cartpole.png",
    )
    engine.utils.closed_loop_plot(
        Z_nom, Z_ol, Z_cl, comp_idx=1, comp_name="pole angle",
        save="figures/example_cartpole_tracking.png",
    )
    print("plots          : figures/example_cartpole.png, figures/example_cartpole_tracking.png")


if __name__ == "__main__":
    main()