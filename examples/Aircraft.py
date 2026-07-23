# ============================================================================
# example_aircraft.py -- 6DOF aircraft: full GRACE suite on a high-dim system
# ============================================================================
# The hardest system in the set: 12 states, 3 controls, stiff nonlinear 6DOF
# flight dynamics. Shows the framework scales to a realistic aircraft -- min-
# effort maneuver, LQR tracking under turbulence, reachability, and dihedral
# codesign -- with the same API as every other system.
#
#   state:   [u,v,w, p,q,r, phi,theta,psi, X,Y,Z]  (body vel, rates, euler, pos)
#   control: [aileron, elevator, rudder]  (degrees)
# ============================================================================
import numpy as np
import os

import grace
from Aircraft_Dynamics import f as aircraft, f_codesign, X0


def main():
    N = 60
    # Lane-change maneuver: veer left then straighten back out, ending in the SAME
    # attitude/velocity as trim but displaced laterally in Y.  The aircraft rolls one
    # way to push sideways, then rolls back to level -- a smooth S to a Y offset.
    # Target the attitude/rate states (return to level) plus the lateral position Y:
    lane_idx = [3, 4, 5, 6, 7, 8, 10]     # p,q,r, phi,theta,psi, Y
    system = grace.build(
        aircraft, nx=12, nu=3, N=N, z0=list(X0), dt=0.05,
        target_idx=lane_idx, pos_idx=(9, 10), job="aircraft",
    )
    engine = grace.GRACE(system)

    target = X0.copy()
    # attitude/rates return to trim (level, no residual rates) -- already trim in X0,
    # so the only change is the lateral offset:
    target[10] = 10.0                     # Y = +10 m  (ended one lane to the left)

    # --- 1. Newton feasibility shoot ---
    U_n = engine.shooting.newton_shoot(target)
    print(f"newton_shoot   : endpoint error {np.linalg.norm(system.endpoint(U_n) - system.target(target)):.2e}")

    # --- 2. Minimum-effort shoot, with control surface deflection limits (box) ---
    U = engine.shooting.lambda_shoot(target, u_lo=[-20, -15, -10], u_hi=[20, 15, 10])
    print(f"lambda_shoot   : endpoint error {np.linalg.norm(system.endpoint(U) - system.target(target)):.2e}, "
          f"cost {float(U @ U):.2f}")

    # --- 3. LQR tracking under turbulence ---
    Q = np.diag([1, 1, 1, 1, 1, 1, 10, 10, 1, 1, 1, 1.0]); R = np.eye(3)
    gains, Z_nom = engine.tracking.lqr_gains(U, Q, R); Z_nom = np.array(Z_nom)
    rng = np.random.default_rng(0)
    disturb = rng.standard_normal((N, 12)) * 0.02
    Z_ol, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=False)
    Z_cl, _ = engine.utils.simulate(U, gains, Z_nom, disturb=disturb, feedback=True)
    err_ol = np.linalg.norm(Z_ol - Z_nom, axis=1).mean()
    err_cl = np.linalg.norm(Z_cl - Z_nom, axis=1).mean()
    print(f"lqr_gains      : mean tracking error  open-loop {err_ol:.3f}  closed-loop {err_cl:.3f}")

    # --- 4. Reachability summary ---
    engine.reachability.print_summary(U, name="aircraft")

    # --- 5. Codesign: optimize the dihedral effect (roll-moment coefficient Clb) ---
    # Trade control effort against a manufacturing penalty for dihedral, on a comparable
    # physical scale so the weight sweep traces a real front:
    clb_base, k_dih = -0.02, 0.006
    def deg_of(clb): return (clb_base - clb) / k_dih
    def clb_of(deg): return clb_base - k_dih * deg

    codesign = grace.Codesign(f_codesign, nx=12, nu=3, N=N, z0=list(X0), dt=0.05)
    U_cd, p_opt, front = codesign.optimize(
        target=target, param_name="Clb", objective=lambda clb: 15.0 * deg_of(clb) ** 2,
        p0=clb_of(10.0), p_bounds=(clb_of(20.0), clb_of(0.0)),
        weights=np.linspace(0, 1, 8), target_idx=lane_idx, plot=True,
    )
    dihedrals = [round(deg_of(fp["param"]), 1) for fp in front]
    print(f"codesign       : optimal dihedral = {deg_of(p_opt):.1f} deg, front sweeps {dihedrals}")

    # --- 6. Obstacle avoidance: a hazard on the centerline forces a symmetric detour ---
    # A longer horizon gives the aircraft room to bank around the hazard and settle back
    # onto the original flight line (Y = 0). The target includes the downrange X so the
    # aircraft is required to return to the centerline rather than end displaced:
    N_obs = 200
    obs_lane_idx = [3, 4, 5, 6, 7, 8, 9, 10]     # include X (9) so downrange is pinned
    sys_obs = grace.build(
        aircraft, nx=12, nu=3, N=N_obs, z0=list(X0), dt=0.05,
        target_idx=obs_lane_idx, pos_idx=(9, 10), job="aircraft_obstacle",
    )
    eng_obs = grace.GRACE(sys_obs)
    downrange = sys_obs.rollout(np.zeros(N_obs * 3))[-1, 9]   # straight-flight distance
    target_obs = X0.copy()
    target_obs[9] = downrange                     # end at the downrange point
    target_obs[10] = 0.0                          # back on the original centerline
    obstacles = [[downrange * 0.5, 0.0]]          # hazard centered on the corridor
    R_obs = 8.0
    U_obs = eng_obs.shooting.lambda_shoot(target_obs, obstacles=obstacles, R=R_obs,
                                          pos_idx=(9, 10),
                                          u_lo=[-20, -15, -10], u_hi=[20, 15, 10])
    Z_obs = sys_obs.rollout(U_obs)
    clearance = min((np.sum((Z_obs[:, [9, 10]] - np.asarray(o)) ** 2, axis=1) ** 0.5).min()
                    for o in obstacles)
    print(f"obstacle_shoot : clearance {clearance:.2f} (need >= {R_obs}), "
          f"endpoint error {np.linalg.norm(sys_obs.endpoint(U_obs) - sys_obs.target(target_obs)):.2e}")

    # --- Plots ---
    os.makedirs("figures", exist_ok=True)
    Z = system.rollout(U)
    engine.utils.plotting(
        Z, U, pos_idx=(9, 10),
        state_names=["u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "X", "Y", "Z"],
        control_names=["aileron", "elevator", "rudder"],
        title="6DOF aircraft -- turn 20 deg then level the wings",
        save="figures/example_aircraft.png",
    )
    engine.utils.plotting(
        Z_obs, U_obs, obstacles=obstacles, R=R_obs, pos_idx=(9, 10),
        state_names=["u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "X", "Y", "Z"],
        control_names=["aileron", "elevator", "rudder"],
        title="6DOF aircraft -- obstacle avoidance",
        save="figures/example_aircraft_obstacles.png",
    )
    engine.utils.closed_loop_plot(
        Z_nom, Z_ol, Z_cl, comp_idx=10, comp_name="lateral position Y",
        save="figures/example_aircraft_tracking.png",
    )
    print("plots          : figures/example_aircraft.png, figures/example_aircraft_obstacles.png, "
          "figures/example_aircraft_tracking.png")


if __name__ == "__main__":
    main()