# ============================================================================
# diagnostics.py -- how optimal and how feasible a control tape is:
# ============================================================================
# The stationarity residual is the KKT condition of the minimum-effort shoot:
# ||2U + Co^T lam|| / ||2U||.  It is zero at an optimal control, so it measures
# how far a solution is from optimal without needing a reference optimizer.
# ============================================================================

import numpy as np


# Compute the stationarity residual of a control tape:
def stationarity(system, U, reg=1e-8):

    # Evaluate the endpoint Jacobian and controllability Gramian:
    U = np.asarray(U).flatten()
    _, Co = system.endpoint_jac(U)
    W = Co @ Co.T + reg * np.eye(system.m)

    # Form the reduced gradient of the cost on the feasible manifold:
    lam = np.linalg.solve(W, -Co @ (2 * U))
    R = 2 * U + Co.T @ lam

    # Return the normalized KKT residual:
    return float(np.linalg.norm(R) / max(np.linalg.norm(2 * U), 1e-9))


# Compute cost, endpoint error, and stationarity for a control tape:
def diagnostics(system, U, target):

    # Reduce the target and evaluate the endpoint:
    U = np.asarray(U).flatten()
    zt = system.target(target)
    e = system.endpoint(U)

    # Return the three solution metrics:
    return dict(cost=float(U @ U),
                endpoint_error=float(np.linalg.norm(e - zt)),
                stationarity=stationarity(system, U))


# Compute the minimum obstacle clearance of a control tape:
def clearance(system, U, obstacles, R, pos_idx=(0, 1)):

    # Position trajectory and its distance to each obstacle:
    p = system.rollout(U)[:, list(pos_idx)]
    OBS = [np.asarray(o, float) for o in obstacles]

    # Return the smallest clearance across all obstacles and nodes:
    return min((np.sum((p - o) ** 2, axis=1) ** 0.5).min() for o in OBS)
