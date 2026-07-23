# ============================================================================
# bounds.py -- control box constraints and R weighting for the shooting solvers:
# ============================================================================
# Two optional refinements shared by newton_shoot and lambda_shoot:
#   box projection  -- clip the control tape to per-control [u_lo, u_hi] bounds
#                      after every step.  Projection (not active-set freezing) is
#                      used deliberately: testing showed freezing controls at a
#                      bound removes them from the least-squares and does worse on
#                      binding cases, while projection respects arbitrary
#                      asymmetric bounds and still reaches reachable targets.
#   R weighting     -- weight the control cost as U^T R U so some controls can be
#                      penalized more than others.  R enters the cost gradient
#                      (2 R U) and the effort metric, letting the optimizer favor
#                      cheaper controls over expensive ones.
# ============================================================================

import numpy as np


# Expand per-control bounds to the full flattened control tape:
def expand_bounds(u_lo, u_hi, N, nu):

    # Default to unbounded when a side is not given:
    lo = np.full(N * nu, -np.inf)
    hi = np.full(N * nu, np.inf)

    # Fill each control period with the per-channel bounds:
    if u_lo is not None:
        lo = np.tile(np.asarray(u_lo, dtype=float), N)
    if u_hi is not None:
        hi = np.tile(np.asarray(u_hi, dtype=float), N)

    return lo, hi


# Project a control tape onto the box (no-op when bounds are absent):
def project_box(U, lo, hi):
    return np.clip(U, lo, hi)


# Build the per-control-vector R weight (diagonal) from a per-channel spec:
def expand_weights(R_weights, N, nu):

    # Default to unit weights (ordinary ||U||^2) when not given:
    if R_weights is None:
        return np.ones(N * nu)

    # Tile the per-channel weights across every control period:
    return np.tile(np.asarray(R_weights, dtype=float), N)


# Weighted control cost U^T R U and its gradient 2 R U:
def weighted_cost(U, r_diag):
    return float(U @ (r_diag * U))


def weighted_cost_grad(U, r_diag):
    return 2.0 * r_diag * U


# Solve A x = b robustly, falling back to least-squares if A is singular.
# Underactuated or over-constrained targets can make the normal-equations matrix
# rank-deficient; lstsq degrades gracefully instead of raising:
def safe_solve(A, b):
    import numpy as np
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]