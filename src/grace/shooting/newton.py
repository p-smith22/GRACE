# ============================================================================
# newton.py -- Newton feasibility shoot:
# ============================================================================
# Drives the endpoint map g(U) onto a target by Gauss-Newton steps on the
# least-norm correction dU = Co^T (Co Co^T)^-1 r.  This reaches a feasible
# trajectory but does not minimize control effort -- that is lambda_shoot.
# ============================================================================

import numpy as np

from .bounds import expand_bounds, project_box, safe_solve


# Shoot the control tape so the endpoint reaches the target:
def newton_shoot(system, z_target, U0=None, it=25, reg=1e-8, u_lo=None, u_hi=None):

    # Reduce the target to the constrained endpoint components:
    zt = system.target(z_target)
    m = system.m

    # Expand optional per-control box bounds to the full tape:
    lo, hi = expand_bounds(u_lo, u_hi, system.N, system.nu)

    # Start from zero controls or a provided warm start, projected into the box:
    U = np.zeros(system.N * system.nu) if U0 is None else np.asarray(U0).flatten().copy()
    U = project_box(U, lo, hi)

    # Take Gauss-Newton steps on the least-norm feasibility correction:
    for _ in range(it):

        # Evaluate the endpoint residual and its Jacobian:
        e, J = system.endpoint_jac(U)
        r = e - zt

        # Stop once the endpoint is reached:
        if np.linalg.norm(r) < 1e-9:
            break

        # Apply the least-norm correction, projected back into the box:
        U = U - J.T @ safe_solve(J @ J.T + reg * np.eye(m), r)
        U = project_box(U, lo, hi)

    # Return the feasible control tape:
    return U