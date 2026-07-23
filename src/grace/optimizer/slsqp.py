# ============================================================================
# slsqp.py -- direct optimization over the compiled system:
# ============================================================================
# Minimizes control effort subject to the endpoint constraint, and optionally
# obstacle-clearance constraints, using SLSQP.  Because the endpoint map and
# its Jacobian are already compiled by CasADi, each optimizer evaluation is
# cheap, so this is an efficient reference solver and the target GRACE's
# shooting solvers are compared against.
# ============================================================================

import numpy as np
from scipy.optimize import minimize


# Optimize the minimum-effort control, optionally avoiding obstacles:
def optimize(system, z_target, obstacles=None, R=None, pos_idx=(0, 1),
             U0=None, maxit=500, ftol=1e-9):

    # Reduce the target and set the warm start:
    zt = system.target(z_target)
    x0 = np.zeros(system.N * system.nu) if U0 is None else np.asarray(U0).flatten()

    # Objective is the control effort with analytic gradient:
    def cost(U):
        return float(U @ U)

    def cost_grad(U):
        return 2 * U

    # The endpoint equality uses the compiled endpoint map:
    def con_end(U):
        return system.endpoint(U) - zt

    constraints = [{"type": "eq", "fun": con_end}]

    # Add obstacle-clearance inequalities when obstacles are given:
    if obstacles is not None:
        pi = list(pos_idx)
        OBS = [np.asarray(o, float) for o in obstacles]

        def con_obs(U):
            p = system.rollout(U)[:, pi]
            return np.concatenate([np.sum((p[1:] - o) ** 2, axis=1) - R ** 2 for o in OBS])

        constraints.append({"type": "ineq", "fun": con_obs})

    # Run SLSQP over the compiled system:
    r = minimize(cost, x0, jac=cost_grad, constraints=constraints,
                 method="SLSQP", options={"maxiter": maxit, "ftol": ftol})

    # Return the optimal control tape:
    return r.x
