# Import packages:
import numpy as np
from .bounds import expand_bounds, project_box, safe_solve

# Newton shoot to drive the endpoint onto the target:
def newton_shoot(system, zt, U0=None, it=40, tol=1e-11,
                 step_max=20.0, reg=1e-8):

    # Extract number of constrained states:
    m = system.m

    # Start from the supplied control, or from rest:
    U = np.zeros(system.N * system.nu) if U0 is None else np.asarray(U0, float).copy()

    # Iterate the least-norm Newton correction on the endpoint residual:
    for _ in range(it):

        # Evaluate the endpoint and its Jacobian:
        e, Co = system.endpoint_jac(U)

        # Ensure it is finite, no reason continuing if its broken:
        if not np.all(np.isfinite(e)):
            break

        # Stop once the endpoint is on target:
        r = e - zt
        rn = np.linalg.norm(r)
        if rn < tol:
            break

        # Least-norm correction that cancels the residual at the linearization:
        dU = -Co.T @ safe_solve(Co @ Co.T + reg * np.eye(m), r)

        # Cap the step for stability:
        n = np.linalg.norm(dU)
        if n > step_max:
            dU = dU * step_max / n

        # Shorten the step until it actually reduces the residual:
        moved = False
        for a in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
            Ut = U + a * dU
            et = system.endpoint(Ut)
            if np.all(np.isfinite(et)) and np.linalg.norm(et - zt) < rn:
                U = Ut
                moved = True
                break

        # Break if no step size can improve:
        if not moved:
            break

    # Return control:
    return U