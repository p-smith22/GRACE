# ============================================================================
# lqr.py -- trajectory-tracking LQR gains:
# ============================================================================
# Linearizes the dynamics about a presolved nominal trajectory and runs the
# backward Riccati recursion to get time-varying feedback gains K_k.  The
# closed-loop law u = u_nom - K (z - z_nom) rejects disturbances to third order
# while staying far cheaper than re-solving the trajectory online.
# ============================================================================

import numpy as np


# Compute time-varying LQR gains along a nominal control tape:
def lqr_gains(system, control, Q, R):

    # Roll the nominal control out to the nominal state trajectory:
    Z = system.rollout(control)
    Un = np.asarray(control).reshape(system.N, system.nu)

    # Cast the weights to arrays:
    Q = np.array(Q, float)
    R = np.array(R, float)

    # Initialize the terminal cost-to-go and gain list:
    P = Q.copy()
    gains = [None] * system.N

    # Run the backward Riccati recursion along the trajectory:
    for k in range(system.N - 1, -1, -1):

        # Linearize the one-step dynamics about the nominal point:
        A, B = system.step_jac(Z[k], Un[k])

        # Solve for the feedback gain and update the cost-to-go:
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        gains[k] = K

    # Return the gains and the nominal trajectory they track:
    return gains, Z
