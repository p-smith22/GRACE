# Import package:
import numpy as np
from scipy.optimize import nnls
from ..shooting.lambda_shoot import compile_constraints, eval_constraints


# Certify how close a control tape is to a KKT point:
def stationarity(system, U, constraints=(), R_weights=None, reg=1e-10,
                 tol_near=1e-2):

    # For an unconstrained shoot, optimality means the cost gradient lies in the
    # row space of the endpoint Jacobian: nothing else can move the endpoint, so
    # anything left over is effort that buys nothing.
    #
    # With constraints the test is not "does the solver's own multiplier balance
    # the gradient" but "does ANY valid multiplier balance it".  Those differ:
    # an augmented Lagrangian can hold a constraint by penalty with its
    # multiplier decayed to zero, which certifies nothing while the trajectory
    # is in fact optimal.  Asking whether a certificate exists is the definition
    # of a KKT point, and it needs no threshold on which rows count active --
    # the sign condition eta >= 0 does that job by itself.
    U = np.asarray(U, float).flatten()
    _, Co = system.endpoint_jac(U)
    Co = np.asarray(Co)

    # Weighted cost gradient, matching whatever the solve minimized:
    r_diag = (np.ones(U.size) if R_weights is None
              else np.tile(np.asarray(R_weights, float).ravel(), system.N))
    g = 2.0 * r_diag * U
    g_norm = max(float(np.linalg.norm(g)), 1e-12)

    # Project out the endpoint rows.  Its multiplier is free in sign, so it can
    # absorb any component in that subspace and only what remains is a residual:
    def project(v):
        return v - Co.T @ np.linalg.solve(Co @ Co.T + reg * np.eye(Co.shape[0]),
                                          Co @ v)

    cons = list(constraints)
    if not cons:
        return float(np.linalg.norm(project(g)) / g_norm)

    h, J = eval_constraints(system, compile_constraints(system, cons), U,
                            want_jac=True)
    if h.size == 0:
        return float(np.linalg.norm(project(g)) / g_norm)

    # Rows that could plausibly be carrying load.  This is only a shortlist: the
    # sign condition below, not this threshold, decides what is really active:
    near = np.where(h > -tol_near)[0]
    if near.size == 0:
        return float(np.linalg.norm(project(g)) / g_norm)

    P_JT = np.column_stack([project(J[i]) for i in near])
    P_g = project(g)

    # Try the closed form first.  A multiplier set only has to exist to certify
    # the point, so if least squares happens to return one that is non-negative
    # it is already a valid certificate and there is nothing to search for:
    eta, *_ = np.linalg.lstsq(P_JT, -P_g, rcond=None)
    if eta.size and eta.min() >= -1e-12:
        return _residual(P_g, P_JT, eta, h[near], g_norm)

    # Otherwise some row wants to pull the wrong way, which means the active set
    # is not what the shortlist assumed.  Which inequalities are active is a
    # search, not a formula -- that is exactly why picking them by a threshold
    # gives a different answer for every threshold -- so solve for the best
    # non-negative multiplier and let the sign condition settle it:
    eta, _ = nnls(P_JT, -P_g)
    return _residual(P_g, P_JT, eta, h[near], g_norm)


# Stationarity residual, charged for any complementarity the fit violated:
def _residual(P_g, P_JT, eta, h_near, g_norm):

    # Stationarity alone is not a KKT certificate.  A multiplier is only allowed
    # to be nonzero on a row that is actually at its limit, so a fit that leans
    # on a row sitting comfortably inside has certified nothing.  Charging that
    # slack into the residual keeps one number honest rather than reporting a
    # small stationarity beside a violated complementarity condition:
    stat = float(np.linalg.norm(P_g + P_JT @ eta))
    slack = float(np.max(eta * np.maximum(0.0, -h_near))) if eta.size else 0.0
    return float((stat + slack) / g_norm)

# Compute cost, endpoint error, stationarity, and worst violation:
def diagnostics(system, U, target, constraints=(), R_weights=None):

    # Reshape control:
    U = np.asarray(U, float).flatten()

    # Only focus on constrained endpoints:
    zt = system.target(target)
    e = system.endpoint(U)

    # Worst constraint violation, positive if a limit was broken:
    cons = list(constraints)
    worst = float("-inf")
    if cons:
        h, _ = eval_constraints(system, compile_constraints(system, cons), U)
        worst = float(h.max()) if h.size else float("-inf")

    r_diag = (np.ones(U.size) if R_weights is None
              else np.tile(np.asarray(R_weights, float).ravel(), system.N))

    # Return the solution metrics:
    return dict(cost=float(U @ (r_diag * U)),
                endpoint_error=float(np.linalg.norm(e - zt)),
                stationarity=stationarity(system, U, cons, R_weights),
                worst_violation=worst)


# Compute the minimum obstacle clearance of a control sequence:
def clearance(system, U, obstacles, R, pos_idx=(0, 1), samples=25):

    # Rollout trajectory of relevant states:
    p = np.asarray(system.rollout(U))[:, list(pos_idx)]
    OBS = [np.asarray(o, float) for o in obstacles]

    # Measure along the path, not just at the nodes.  A fast trajectory can pass
    # every node test and still cut the corner between two of them, so the nodal
    # minimum reports clear on a trajectory that is not:
    worst = np.inf
    for o in OBS:
        for k in range(len(p) - 1):
            for t in np.linspace(0.0, 1.0, samples):
                worst = min(worst, np.linalg.norm((1.0 - t) * p[k] + t * p[k + 1] - o))
    return float(worst)