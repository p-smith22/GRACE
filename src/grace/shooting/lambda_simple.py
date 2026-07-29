# Import packages:
import numpy as np
from .newton import newton_shoot
from .bounds import expand_bounds, project_box, expand_weights, weighted_cost, weighted_cost_grad, safe_solve

# Newton shoot for optimal control sequence:
def lambda_simple(system, z_target, U0=None, max_it=60, ftol=1e-2, tr0=0.3, reg=1e-8,
                  u_lo=None, u_hi=None, R_weights=None):

    # Only focus on constrained states (i.e., ones we are focusing on for endpoint):
    zt = system.target(z_target)

    # Extract number of constriained states:
    m = system.m

    # Newton shoot for feasible control sequence:
    U = newton_shoot(system, zt, U0, u_lo=u_lo, u_hi=u_hi)

    # Expand bounds to full trajectory, build weight matrix:
    lo, hi = expand_bounds(u_lo, u_hi, system.N, system.nu)
    r_diag = expand_weights(R_weights, system.N, system.nu)

    # Track the current weighted cost and trust-region size:
    c = weighted_cost(U, r_diag)
    tr = tr0

    # Begin lambda shoot was max iterations:
    for _ in range(max_it):

        # Evaluate the endpoint and its Jacbian:
        e, Co = system.endpoint_jac(U)

        # Construct the controllability matrix:
        W = Co @ Co.T + reg * np.eye(m)

        # Compute residual on Lagrange multiplier:
        gc = weighted_cost_grad(U, r_diag)
        lam = safe_solve(W, -Co @ gc)
        R = gc + Co.T @ lam

        # Stop once the reduced gradient is small relative to the cost gradient:
        if np.linalg.norm(R) / max(np.linalg.norm(gc), 1e-9) < ftol:
            break

        # Compute descent direction (steepest descent, normalized):
        d = -R / np.linalg.norm(R)

        # Compute controls that need to be adjusted:
        d = d - Co.T @ safe_solve(W, Co @ d)

        # Adjust control:
        Ut = U + tr * np.linalg.norm(U) * d

        # Project the trial step into the box constraints:
        Ut = project_box(Ut, lo, hi)

        # Recover feasibility in the endpoint via newton shoot methodology (intermediate:
        newton_shoot(system, zt, Ut, u_lo=u_lo, u_hi=u_hi, it=3, tol=1e-3)

        # Ensure new step lowers cost and is feasible:
        ct = weighted_cost(Ut, r_diag)
        if ct < c and np.linalg.norm(system.endpoint(Ut) - zt) < 5e-3:
            U = Ut
            c = ct
            tr = min(tr * 1.3, 1.0)

        # Otherwise shrink the trust region:
        else:
            tr *= 0.5

        # Stop if the trust region collapses:
        if tr < 1e-4:
            break

    # Clean up feasibility:
    U = newton_shoot(system, zt, U, u_lo=u_lo, u_hi=u_hi)

    # Flag if the endpoint could not be reached:
    ee = np.linalg.norm(system.endpoint(U) - zt)
    system._infeasible = bool(ee > 1e-2)
    if system._infeasible:
        print("[GRACE] WARNING: target does not appear reachable. "
              "Try a longer horizon N, more time, or a closer target.")

    # Return control:
    return U
