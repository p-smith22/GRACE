# GRACE Documentation

**GRACE** (Gramian-based Reachability Analysis and Control Engine) solves nonlinear
optimal control problems by controllability-Gramian shooting on CasADi-compiled
analytic derivatives. Provide a dynamics function and GRACE gives you the control
solution to any reachable state.

## Contents

- [Getting started](getting_started.md) --- build a system, run the engine
- [Shooting](shooting.md) --- `newton_shoot`, `lambda_shoot` (simple and obstacle modes)
- [Tracking](tracking.md) --- LQR gains and closed-loop simulation
- [Reachability](reachability.md) --- controllability metrics and reports
- [Optimizer](optimizer.md) --- the reference direct optimizer
- [Codesign](codesign.md) --- joint control-and-parameter optimization
- [Utilities](utils.md) --- simulate, plotting, diagnostics, comparison
- [Caching](caching.md) --- saving and reloading compiled graphs by job name

## The core idea

Every GRACE solver runs on one compiled object: the endpoint map `g(U)` that sends a
control tape `U` to the final state, together with its analytic Jacobian `Co = dg/dU`.
The controllability Gramian `W = Co Co^T` governs reachability --- which directions the
endpoint moves in cheaply, and which are expensive. Shooting solves the minimum-effort
control by projected-gradient descent on this manifold; obstacle avoidance adds hard
path constraints; reachability reads the structure of `W` directly.

## Minimal example

```python
import grace
import casadi as ca
import numpy as np

# Define dynamics as CasADi expressions:
def dynamics(x, u):
    return ca.vertcat(x[2], x[3], u[0], u[1])

# Build a system and engine:
system = grace.build(dynamics, nx=4, nu=2, N=80, z0=[0, 0, 0, 0], dt=0.1)
engine = grace.GRACE(system)

# Solve the minimum-effort control to a target:
control = engine.shooting.lambda_shoot(np.array([10.0, 0, 0, 0]))
```
