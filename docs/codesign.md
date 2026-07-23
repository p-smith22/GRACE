# Codesign

Joint control-and-parameter optimization. You name a design parameter that appears in the
dynamics; GRACE builds an extra rollout carrying the endpoint's sensitivity to it, then
alternates an inner lambda shoot with an outer design step. The design gradient comes free
from the envelope theorem --- at the inner optimum only the explicit parameter dependence
matters, so no differentiation through the inner solve is needed. Sweeping the design
weight traces a Pareto front of control effort against a design objective.

## Usage

The dynamics function takes an extra parameter argument `(x, u, p)`:

```python
import numpy as np, casadi as ca
from grace import Codesign

# Dynamics with a named design parameter p:
def dynamics(x, u, p):
    return ca.vertcat(x[2], x[3], p * u[0], p * u[1])

# Design objective to trade against control effort:
def objective(p):
    return p ** 2

cd = Codesign(dynamics, nx=4, nu=2, N=60, z0=[0, 0, 0, 0], dt=0.1)
control, param, front = cd.optimize(target, param_name="g", objective=objective,
                                    p0=1.0, p_bounds=(0.5, 2.0),
                                    weights=None, job="codesign")
```

- `param_name` --- the symbol name of the design parameter
- `objective(p)` --- the design cost to trade against control effort
- `p0`, `p_bounds` --- initial design value and its bounds
- `weights` --- the sweep of trade weights that traces the Pareto front
- returns the selected control, the selected parameter, and the full front

A Pareto figure is written to the figures directory. The returned control is chosen at a
balanced trade weight; the full front lets you pick another operating point.
