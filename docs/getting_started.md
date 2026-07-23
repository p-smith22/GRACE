# Getting started

## Install

```bash
git clone https://github.com/<username>/grace.git
cd grace
pip install -e .
```

## Build a system

`grace.build` compiles everything the engine needs from your dynamics function by
RK4 integration and automatic differentiation.

```python
system = grace.build(dynamics, nx, nu, N, z0, dt,
                     target_idx=None, pos_idx=None, substeps=1, jit=False, job=None)
```

- `dynamics(x, u)` --- returns the continuous-time state derivative as CasADi expressions
- `nx`, `nu` --- state and control dimensions
- `N` --- number of horizon steps
- `z0` --- initial state
- `dt` --- time step (so the horizon covers `N * dt` seconds)
- `target_idx` --- which endpoint components the shoot constrains (default all)
- `pos_idx` --- which two state components are position (required for obstacle avoidance)
- `substeps` --- RK4 substeps per control period (raise for stiff dynamics)
- `jit` --- JIT-compile the graph for steady-state speed
- `job` --- a name used for graph caching (see caching.md)

## Run the engine

```python
engine = grace.GRACE(system)
```

The engine exposes solver groups as namespaces:

- `engine.shooting` --- newton and lambda shooting
- `engine.tracking` --- LQR gains
- `engine.reachability` --- controllability analysis
- `engine.optimizer` --- the reference direct optimizer
- `engine.utils` --- simulate, plotting, diagnostics, comparison
