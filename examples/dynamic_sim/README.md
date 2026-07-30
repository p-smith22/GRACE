# Interactive simulation demo

A real-time showcase of GRACE: every trajectory the robot flies is a GRACE
minimum-effort solve, tracked by a finite-horizon LQR controller.

Controls:

| input | action |
|---|---|
| left click | set a goal -- GRACE plans a minimum-effort trajectory to it |
| right drag | draw a circular obstacle |
| `c` | clear obstacles |
| `r` | reset the drone |
| `esc` | quit |
