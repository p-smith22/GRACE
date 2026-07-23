# ============================================================================
# lambda_simple.py -- minimum-effort optimal shoot to a target:
# ============================================================================
# Projected-gradient descent on the control cost U^T R U confined to the
# feasible manifold g(U) = target.  The descent direction is the reduced
# gradient R = 2 R U + Co^T lam, reprojected onto the manifold each step.
# Optional per-control box bounds [u_lo, u_hi] (enforced by projection) and an
# R weighting (per-control cost weights) let some controls be favored over
# others.  This is the simple (no-obstacle) branch of lambda_shoot.
# ============================================================================

import numpy as np

from .newton import newton_shoot
from .bounds import expand_bounds, project_box, expand_weights, weighted_cost, weighted_cost_grad, safe_solve


# Shoot the minimum-effort control tape that reaches the target:
def lambda_simple(system, z_target, U0=None, max_it=60, ftol=1e-2, tr0=0.3, reg=1e-8,
                  u_lo=None, u_hi=None, R_weights=None):

    # Reduce the target and warm-start from a feasible (box-projected) Newton shot:
    zt = system.target(z_target)
    m = system.m
    U = newton_shoot(system, zt, U0, u_lo=u_lo, u_hi=u_hi)

    # Expand optional per-control box bounds and R cost weights:
    lo, hi = expand_bounds(u_lo, u_hi, system.N, system.nu)
    r_diag = expand_weights(R_weights, system.N, system.nu)

    # Track the current weighted cost and trust-region size:
    c = weighted_cost(U, r_diag)
    tr = tr0

    # Descend the weighted control cost within the feasible manifold:
    for _ in range(max_it):

        # Evaluate the endpoint Jacobian and controllability Gramian:
        e, Co = system.endpoint_jac(U)
        W = Co @ Co.T + reg * np.eye(m)

        # Form the reduced gradient of the weighted cost on the manifold:
        gc = weighted_cost_grad(U, r_diag)
        lam = safe_solve(W, -Co @ gc)
        R = gc + Co.T @ lam

        # Stop once the reduced gradient is small relative to the cost gradient:
        if np.linalg.norm(R) / max(np.linalg.norm(gc), 1e-9) < ftol:
            break

        # Descend along the tangent (nullspace) direction:
        d = -R / np.linalg.norm(R)
        d = d - Co.T @ safe_solve(W, Co @ d)
        Ut = U + tr * np.linalg.norm(U) * d

        # Project the trial step into the box:
        Ut = project_box(Ut, lo, hi)

        # Reproject the trial step back onto the feasible manifold (staying in the box):
        for _ in range(3):
            rr = system.endpoint(Ut) - zt
            if np.linalg.norm(rr) < 1e-3:
                break
            _, Jt = system.endpoint_jac(Ut)
            Ut = Ut - Jt.T @ safe_solve(Jt @ Jt.T + reg * np.eye(m), rr)
            Ut = project_box(Ut, lo, hi)

        # Accept the step if it lowers the weighted cost and stays feasible:
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

    # Clean up feasibility and return the optimal control tape:
    U = newton_shoot(system, zt, U, u_lo=u_lo, u_hi=u_hi)

    # Flag if the endpoint could not be reached -- an impossible target for this horizon
    # (too little time) or an underactuated system that cannot hit every target state:
    ee = np.linalg.norm(system.endpoint(U) - zt)
    system._infeasible = bool(ee > 1e-2)
    if system._infeasible:
        print("[grace] warning: target does not appear reachable for this horizon "
              "(endpoint error %.2e). Try a longer horizon N, more time, or a closer target." % ee)

    return U