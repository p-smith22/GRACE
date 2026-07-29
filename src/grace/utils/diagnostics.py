# Import package:
import numpy as np


# Compute the stationarity residual of a control tape:
def stationarity(system, U, reg=1e-8):

    # Reshape control:
    U = np.asarray(U).flatten()

    # Evaluate the endpoint Jacobian:
    _, Co = system.endpoint_jac(U)

    # Compute Gramian:
    W = Co @ Co.T + reg * np.eye(system.m)

    # Compute Lagrange multiplier residual:
    lam = np.linalg.solve(W, -Co @ (2 * U))
    R = 2 * U + Co.T @ lam

    # Return normalized residual:
    return float(np.linalg.norm(R) / max(np.linalg.norm(2 * U), 1e-9))

# Compute cost, endpoint error, and stationarity for a control tape:
def diagnostics(system, U, target):

    # Reshape control:
    U = np.asarray(U).flatten()

    # Only focus on constrained endpoints:
    zt = system.target(target)

    # Compute endpoint:
    e = system.endpoint(U)

    # Return the three solution metrics:
    return dict(cost=float(U @ U),
                endpoint_error=float(np.linalg.norm(e - zt)),
                stationarity=stationarity(system, U))


# Compute the minimum obstacle clearance of a control sequence:
def clearance(system, U, obstacles, R, pos_idx=(0, 1)):

    # Rollout trajectory of relevant states:
    p = system.rollout(U)[:, list(pos_idx)]

    # Package obstacles:
    OBS = [np.asarray(o, float) for o in obstacles]

    # Return the smallest clearance across all obstacles and nodes:
    return min((np.sum((p - o) ** 2, axis=1) ** 0.5).min() for o in OBS)
