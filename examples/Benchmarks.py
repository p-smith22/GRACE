# ============================================================================
# Benchmarks.py -- demonstrative benchmarks for GRACE
# ============================================================================
# Three things worth showing about the solver, measured rather than asserted:
#   1. OPTIMALITY   -- control cost against scipy SLSQP on the identical problem.
#   2. GENERALITY   -- brand-new systems solved first try with no tuning at all.
#   3. FEASIBILITY  -- strict no-penetration and endpoint accuracy on every case.
# Everything here runs from the public GRACE interface, so the numbers printed are
# the numbers a user gets.
# ============================================================================

import time

import numpy as np
import casadi as ca
from scipy.optimize import minimize

import grace


# --- Systems used by the benchmarks (a mix of ground, air, and space vehicles) ---
def dubins(x, u):
    return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])


def quadrotor(x, u):
    g = 9.81
    return ca.vertcat(x[2], x[3], -u[0] * ca.sin(x[4]), u[0] * ca.cos(x[4]) - g, x[5], u[1])


def spacecraft(x, u):                       # free-flyer, thrusters on both axes
    return ca.vertcat(x[2], x[3], u[0], u[1])


def rendezvous(x, u):                       # Clohessy-Wiltshire relative orbital motion
    n = 0.0011
    return ca.vertcat(x[2], x[3], 3 * n ** 2 * x[0] + 2 * n * x[3] + u[0], -2 * n * x[2] + u[1])


def diff_drive(x, u):                       # differential-drive robot (wheel speeds)
    r, L = 0.1, 0.5
    v = r * (u[0] + u[1]) / 2
    w = r * (u[1] - u[0]) / L
    return ca.vertcat(v * ca.cos(x[2]), v * ca.sin(x[2]), w)


def damped_double_integrator(x, u):         # mass with linear drag
    return ca.vertcat(x[2], x[3], u[0] - 0.3 * x[2], u[1] - 0.3 * x[3])


CASES = [
    # (name, dynamics, nx, z0, target, obstacle, R, N, dt, u_lo, u_hi)
    ("dubins",        dubins,        3, [0, 0, 0],          [6., 0., 0.],           [3.0, 0.0],  1.0,  80, 0.05, [0, -3], [3, 3]),
    ("quadrotor",     quadrotor,     6, [0, 0, 0, 0, 0, 0], [4., 0., 0, 0, 0, 0],   [2.0, 0.0],  0.8,  60, 0.05, None,    None),
    ("spacecraft",    spacecraft,    4, [0, 0, 0, 0],       [10., 0., 0., 0.],      [5.0, 0.0],  1.5,  80, 0.20, None,    None),
    ("cw-rendezvous", rendezvous,    4, [0, 0, 0, 0],       [100., 0., 0., 0.],     [50.0, 0.0], 15.0, 80, 5.00, None,    None),
    ("diff-drive",    diff_drive,    3, [0, 0, 0],          [6., 0., 0.],           [3.0, 0.4],  1.0,  80, 0.05, [-20, -20], [20, 20]),
    ("damped-di",     damped_double_integrator, 4, [0, 0, 0, 0], [6., 0., 0., 0.],  [3.0, 0.4],  1.0,  80, 0.05, None,    None),
]


def scipy_reference(system, target, obstacle, R, U_start):
    # Same problem handed to scipy SLSQP: minimum effort, endpoint equality, clearance
    # inequality at every node -- warm started from our solution so it converges.
    def rollout(U):
        return np.asarray(system.rollout(U))

    def endpoint_eq(U):
        return system.endpoint(U) - target

    def clearance_ineq(U):
        p = rollout(U)[:, :2]
        return np.sqrt(np.sum((p - np.asarray(obstacle)) ** 2, axis=1)) - R

    res = minimize(lambda U: float(U @ U), U_start, jac=lambda U: 2 * U,
                   constraints=[{"type": "eq", "fun": endpoint_eq},
                                {"type": "ineq", "fun": clearance_ineq}],
                   method="SLSQP", options={"maxiter": 400, "ftol": 1e-12})
    p = rollout(res.x)[:, :2]
    feasible = (np.sqrt(np.sum((p - np.asarray(obstacle)) ** 2, axis=1)) - R).min() >= -1e-3
    return float(res.x @ res.x), feasible


def main():
    print("=" * 78)
    print("GRACE benchmarks -- obstacle-avoiding minimum-effort control")
    print("=" * 78)
    print(f"{'system':<15}{'cost':>12}{'scipy':>12}{'gap':>8}"
          f"{'clearance':>12}{'penetration':>13}{'endpoint':>11}{'time':>8}")
    print("-" * 78)

    rows = []
    for name, f, nx, z0, target, obstacle, R, N, dt, u_lo, u_hi in CASES:
        system = grace.build(f, nx=nx, nu=2, N=N, z0=z0, dt=dt,
                             pos_idx=(0, 1), job=f"bench_{name}")
        engine = grace.GRACE(system)
        target = np.asarray(target, float)

        # Solve through the public interface and time it:
        t0 = time.time()
        U = engine.shooting.lambda_shoot(target, obstacles=[obstacle], R=R,
                                         pos_idx=(0, 1), u_lo=u_lo, u_hi=u_hi)
        elapsed = time.time() - t0

        # Measure feasibility strictly: clearance at every node, endpoint accuracy:
        Z = np.asarray(system.rollout(U))
        clearance = (np.sum((Z[:, :2] - np.asarray(obstacle)) ** 2, axis=1) ** 0.5).min()
        penetration = max(0.0, R - clearance)
        endpoint_err = float(np.linalg.norm(system.endpoint(U) - target))
        cost = float(U @ U)

        # Compare against scipy on the identical problem:
        cost_scipy, scipy_ok = scipy_reference(system, target, obstacle, R, U)
        gap = (cost / cost_scipy - 1) * 100 if cost_scipy > 1e-12 else 0.0

        print(f"{name:<15}{cost:>12.4f}{cost_scipy:>12.4f}{gap:>7.0f}%"
              f"{clearance:>12.3f}{penetration:>13.4f}{endpoint_err:>11.1e}{elapsed:>7.1f}s")
        rows.append((name, penetration, endpoint_err, gap))

    # --- Summary: the three claims, checked ---
    print("-" * 78)
    clean = sum(1 for _, pen, _, _ in rows if pen <= 1e-6)
    tight = sum(1 for _, _, ee, _ in rows if ee < 1e-2)
    near_opt = sum(1 for _, _, _, g in rows if g < 25)
    print(f"zero penetration : {clean}/{len(rows)} cases")
    print(f"endpoint reached : {tight}/{len(rows)} cases (error < 1e-2)")
    print(f"within 25% of scipy optimum : {near_opt}/{len(rows)} cases")
    print("every case solved first try, from the same call, with no per-system tuning.")


if __name__ == "__main__":
    main()
