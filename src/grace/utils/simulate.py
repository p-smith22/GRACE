# Import package:
import numpy as np

# Simulate the system forward under a control sequence (closed- or open-loop)
def simulate(system, control, gains=None, nominal=None, disturb=None, feedback=True):

    # Reshape the control sequence to be (tsteps, n_controls):
    Un = np.asarray(control).reshape(system.N, system.nu)

    # Start from the system's initial state:
    z = system.z0.copy()

    # Initialize trajectory:
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
