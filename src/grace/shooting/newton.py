# Import necessary packages:
import numpy as np
from .bounds import expand_bounds, project_box, safe_solve

# Newton shoot function:
def newton_shoot(system, z_target, U0=None, it=25, reg=1e-8, tol=1e-9, u_lo=None, u_hi=None):

    # Only pull constrained states:
    zt = system.target(z_target)

    # Define number of constrained states:
    m = system.m

    # Stretch bounds to entire timespan:
    lo, hi = expand_bounds(u_lo, u_hi, system.N, system.nu)

    # Use provided warmstart or defualt to zeros if none is given:
    U = np.zeros(system.N * system.nu) if U0 is None else np.asarray(U0).flatten().copy()

    # Ensure controls fit within bounds, clip if not:
    U = project_box(U, lo, hi)

    # Attempt to converge endpoint with a max number of iterations:
    for _ in range(it):

        # Evaluate the endpoint residual and its Jacobian:
        e, J = system.endpoint_jac(U)
        r = e - zt

        # Stop once the endpoint residual is low enough:
        if np.linalg.norm(r) < tol:
            break

        # Apply the least-norm correction, projected back into the box:
        U = U - J.T @ safe_solve(J @ J.T + reg * np.eye(m), r)
        U = project_box(U, lo, hi)

    # Return the feasible control tape:
    return U