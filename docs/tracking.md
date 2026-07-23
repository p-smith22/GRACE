# Tracking

Closed-loop trajectory tracking. Design LQR gains about a nominal, then simulate under
disturbance with `engine.utils.simulate`.

## lqr_gains

Linearizes the dynamics about the nominal trajectory and runs the backward Riccati
recursion for time-varying gains.

```python
gains, nominal = engine.tracking.lqr_gains(control, Q, R)
```

- `control` --- the nominal control tape to track
- `Q` --- state-error weight
- `R` --- control-effort weight
- returns the per-step gains and the nominal state trajectory they track

## Simulating the closed loop

```python
trajectory, applied = engine.utils.simulate(control, gains=gains, nominal=nominal,
                                            disturb=disturbances, feedback=True)
```

With `gains` and `nominal` supplied the law is `u = u_nom - K (z - z_nom)`; without them
the tape plays open loop. See utils.md.
