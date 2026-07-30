# GRACE

**GRACE** (Gramian-based Reachability Analysis and Control Engine) is a lightweight Python
framework for solving nonlinear optimal control problems using controllability Gramian
methods and analytic derivatives generated with CasADi.

Provide a dynamics function and GRACE gives you the control solution to any reachable state:
minimum-effort trajectories, obstacle avoidance, closed-loop tracking, controllability
analysis, and joint control-and-parameter codesign.

## Install

```bash
git clone https://github.com/p-smith22/grace.git
cd grace
pip install -e .
```

## Example

```python
import grace
import numpy as np
import casadi as ca

# Define system dynamics:
def dynamics(x, u):
    return ca.vertcat(x[2], x[3], u[0], u[1])

# Build a CasADi system with given dynamics:
system = grace.build(dynamics, nx=4, nu=2, N=80, z0=[0, 0, 0, 0], dt=0.1, pos_idx=(0, 1))

# Initialize the GRACE engine:
engine = grace.GRACE(system)

# Minimum-effort control to a target:
control = engine.shooting.lambda_shoot(np.array([10.0, 0, 0, 0]))

# Obstacle-avoidance trajectory (same call, with obstacles):
control = engine.shooting.lambda_shoot(np.array([14.0, 0, 0, 0]),
                                       obstacles=[[7.0, 0.0]], R=2.0, pos_idx=(0, 1))

# Controllability analysis:
report = engine.reachability.summary(control)

# Closed-loop tracking:
gains, nominal = engine.tracking.lqr_gains(control, np.diag([10, 10, 1, 1.0]), np.eye(2) * 0.5)
trajectory, applied = engine.utils.simulate(control, gains=gains, nominal=nominal)

# Plot:
engine.utils.plotting(trajectory, control)
```

## Documentation

See `docs/` for per-module documentation, or start with `docs/getting_started.md`.

## Verification

Run the standalone benchmark suite:

```bash
python -m tests.test_benchmarks
```
