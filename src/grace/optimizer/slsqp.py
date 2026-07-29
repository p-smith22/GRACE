# Import packages:
import numpy as np
from scipy.optimize import minimize

# Optimize trajectory with or without obstacles:
def optimize(system, z_target, obstacles=None, R=None, pos_idx=(0, 1),
             U0=None, maxit=500, ftol=1e-9):

    # Reduce the target to constrained states:
    zt = system.target(z_target)

    # Set initial control guess or default to zero if none is provided:
    x0 = np.zeros(system.N * system.nu) if U0 is None else np.asarray(U0).flatten()

    # Define minimum effort objective:
    def cost(U):
        return float(U @ U)

    # Define gradient:
    def cost_grad(U):
        return 2 * U

    # Define endpoint constraint:
    def con_end(U):
        return system.endpoint(U) - zt
    constraints = [{"type": "eq", "fun": con_end}]

    # Add obstacle inequalities if given:
    if obstacles is not None:

        # Package obstacles:
        pi = list(pos_idx)
        OBS = [np.asarray(o, float) for o in obstacles]

        # Set obstacle constraint:
        def con_obs(U):
            p = system.rollout(U)[:, pi]
            return np.concatenate([np.sum((p[1:] - o) ** 2, axis=1) - R ** 2 for o in OBS])
        constraints.append({"type": "ineq", "fun": con_obs})

    # Run SLSQP:
    r = minimize(cost, x0, jac=cost_grad, constraints=constraints,
                 method="SLSQP", options={"maxiter": maxit, "ftol": ftol})

    # Return the optimal control sequence:
    return r.x
