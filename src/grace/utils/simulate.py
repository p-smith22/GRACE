# ============================================================================
# simulate.py -- roll the system forward, open or closed loop:
# ============================================================================
# simulate is multi-use.  With no gains it plays the control tape open loop.
# With tracking gains and a nominal it corrects deviations with feedback
# u = u_nom - K (z - z_nom).  Disturbances are added to the state each step so
# open-loop drift and closed-loop rejection can be compared on the same noise.
# ============================================================================

import numpy as np


# Simulate the system forward under a control tape:
def simulate(system, control, gains=None, nominal=None, disturb=None, feedback=True):

    # Reshape the control tape to per-step controls:
    Un = np.asarray(control).reshape(system.N, system.nu)

    # Start from the system's initial state:
    z = system.z0.copy()
    Z = [z.copy()]

    # Step through the horizon:
    U_applied = []
    for k in range(system.N):

        # Start from the nominal control for this step:
        u = Un[k].copy()

        # Apply tracking feedback when gains and a nominal are provided:
        if feedback and gains is not None and nominal is not None:
            u = u - gains[k] @ (z - nominal[k])

        # Add any disturbance to the state after the step:
        w = disturb[k] if disturb is not None else np.zeros(system.nx)
        z = system.step_np(z, u) + w

        # Record the applied control and new state:
        U_applied.append(u)
        Z.append(z.copy())

    # Return the trajectory and the controls actually applied:
    return np.array(Z), np.array(U_applied)
