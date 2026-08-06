# Import package:
import numpy as np
from ..shooting.lambda_shoot import compile_constraints, eval_constraints

# Compute the stationarity residual of a control tape:
def stationarity(system, U, constraints=(), reg=1e-8, tol_act=1e-4):

    # Reshape control:
    U = np.asarray(U).flatten()

    # Evaluate the endpoint Jacobian:
    _, Co = system.endpoint_jac(U)
    A = Co

    # Evaluate active constraints:
    cons = list(constraints)
    if cons:
        con = compile_constraints(system, cons)
        h, J = eval_constraints(system, con, U, want_jac=True)
        act = np.where(h > -tol_act)[0]
        if act.size:
            A = np.vstack([Co, J[act]])

    # Compute Lagrange multiplier residual:
    g = 2 * U
    W = A @ A.T + reg * np.eye(A.shape[0])
    lam = np.linalg.solve(W, -A @ g)
    R = g + A.T @ lam

    # Return normalized residual:
    return float(np.linalg.norm(R) / max(np.linalg.norm(g), 1e-9))

# Compute cost, endpoint error, stationarity, and worst violation:
def diagnostics(system, U, target, constraints=()):

    # Reshape control:
    U = np.asarray(U).flatten()

    # Only focus on constrained endpoints:
    zt = system.target(target)

    # Compute endpoint:
    e = system.endpoint(U)

    # Worst constraint violation, positive if a limit was broken:
    cons = list(constraints)
    worst = float("-inf")
    if cons:
        h, _ = eval_constraints(system, compile_constraints(system, cons), U)
        worst = float(h.max()) if h.size else float("-inf")

    # Return the solution metrics:
    return dict(cost=float(U @ U),
                endpoint_error=float(np.linalg.norm(e - zt)),
                stationarity=stationarity(system, U, cons),
                worst_violation=worst)


# Compute the minimum obstacle clearance of a control sequence:
def clearance(system, U, obstacles, R, pos_idx=(0, 1), samples=25):

    # Rollout trajectory of relevant states:
    p = np.asarray(system.rollout(U))[:, list(pos_idx)]

    # Package obstacles:
    OBS = [np.asarray(o, float) for o in obstacles]

    # Measure along the path:
    worst = np.inf
    for o in OBS:
        for k in range(len(p) - 1):
            for t in np.linspace(0.0, 1.0, samples):
                worst = min(worst, np.linalg.norm((1.0 - t) * p[k] + t * p[k + 1] - o))
    return float(worst)