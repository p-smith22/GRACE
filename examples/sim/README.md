# Interactive drone demo

A real-time showcase of GRACE: every trajectory the drone flies is a GRACE
minimum-effort solve, tracked by a finite-horizon LQR controller.

    pip install pygame
    python drone_sim.py

Controls:

| input | action |
|---|---|
| left click | set a goal -- GRACE plans a minimum-effort trajectory to it |
| right drag | draw a circular obstacle |
| `c` | clear obstacles |
| `r` | reset the drone |
| `esc` | quit |

Planning runs on a worker thread, so the simulation keeps running while a solve
is in flight; the new trajectory is swapped in as soon as it is ready. If a
request cannot be met (goal unreachable in the horizon, or an obstacle that
cannot be cleared), GRACE reports it and the drone holds position.
