# Utilities

## simulate

Rolls the system forward, open or closed loop.

```python
trajectory, applied = engine.utils.simulate(control, gains=None, nominal=None,
                                            disturb=None, feedback=True)
```

With `gains` and `nominal` supplied it applies `u = u_nom - K (z - z_nom)`; without them it
plays the tape open loop. `disturb` adds a per-step state disturbance so open-loop drift
and closed-loop rejection can be compared on the same noise.

## plotting

One figure with the trajectory (and obstacles), each state, and each control in its own
panel, sharing the time axis.

```python
engine.utils.plotting(trajectory, control, obstacles=None, R=None, pos_idx=(0, 1),
                     state_names=None, control_names=None, title="...", save="...",
                     show_traj=True)
```

## diagnostics

```python
d = engine.utils.diagnostics(control, target)   # cost, endpoint_error, stationarity
s = engine.utils.stationarity(control)          # KKT residual (0 at optimal)
c = engine.utils.clearance(control, obstacles, R, pos_idx=(0, 1))
```

Stationarity is `||2U + Co^T lam|| / ||2U||`, the KKT residual of the minimum-effort
problem. It is zero at an optimal control, so it measures how optimal a solution is with
no reference optimizer.

## compare

Runs the shoot and the optimizer on the same problem, times both, and prints a standard
line per method (method, mode, cost, stationarity, endpoint error, time, clearance).

```python
result = engine.utils.compare(target, obstacles=None, R=None, pos_idx=(0, 1),
                             verbose=True, plot="figures/compare.png")
```

`plot` saves a four-metric panel: compute time, control cost, endpoint error, stationarity.
