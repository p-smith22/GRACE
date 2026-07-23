# ============================================================================
# systems.py -- small verified systems the benchmarks run on:
# ============================================================================
# Easy, well-understood systems so a benchmark failure points at the solver,
# not the model.  These are not meant to be fast, just correct and repeatable.
# ============================================================================

import numpy as np
import casadi as ca

import grace


# A 2D double integrator (linear, four states, two controls):
def double_integrator(N=80, T=8.0):

    # Point-mass dynamics: position integrates velocity, velocity integrates control:
    def dynamics(x, u):
        return ca.vertcat(x[2], x[3], u[0], u[1])

    # Build the system with position indices for obstacle avoidance:
    return grace.build(dynamics, 4, 2, N, [0, 0, 0, 0.0], T / N, pos_idx=(0, 1))


# A cart-pole (nonlinear, four states, one control):
def cart_pole(N=40, T=2.0):

    # Standard cart-pole with a light pole:
    g = 9.81
    mp = 0.2
    L = 0.5

    # Continuous dynamics with the pole hanging down at theta = pi:
    def dynamics(x, u):
        th = x[2]
        s = ca.sin(th)
        c = ca.cos(th)
        den = 1 + mp * s ** 2
        return ca.vertcat(
            x[1],
            (u[0] + mp * s * (L * x[3] ** 2 + g * c)) / den,
            x[3],
            (-u[0] * c - mp * L * x[3] ** 2 * s * c - (1 + mp) * g * s) / (L * den))

    # Build with substeps for the stiff nonlinear dynamics:
    return grace.build(dynamics, 4, 1, N, [0, 0, np.pi, 0.0], T / N, substeps=8)
