# Import packages:
import numpy as np

# Expand bounds to full trajectory:
def expand_bounds(u_lo, u_hi, N, nu):

    # Default to unbounded:
    lo = np.full(N * nu, -np.inf)
    hi = np.full(N * nu, np.inf)

    # Add bounds if provided:
    if u_lo is not None:
        lo = np.tile(np.asarray(u_lo, dtype=float), N)
    if u_hi is not None:
        hi = np.tile(np.asarray(u_hi, dtype=float), N)

    # Return stretched bounds:
    return lo, hi

# Clip control to box constraints:
def project_box(U, lo, hi):
    return np.clip(U, lo, hi)

# Build weight matrix R to weigh controls:
def expand_weights(R_weights, N, nu):

    # Default to identity if note are given:
    if R_weights is None:
        return np.ones(N * nu)

    # If given, tile the per-channel weights across every control period:
    return np.tile(np.asarray(R_weights, dtype=float), N)

# Computed weighted cost with weight matrix R:
def weighted_cost(U, r_diag):
    return float(U @ (r_diag * U))

# Define gradient of weighted cost function:
def weighted_cost_grad(U, r_diag):
    return 2.0 * r_diag * U

# Safe solve function for linear system of form Ax=b:
def safe_solve(A, b):

    # Try to solve linear system normally:
    try:
        return np.linalg.solve(A, b)

    # Least squares fallback to avoid throwing errors:
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]