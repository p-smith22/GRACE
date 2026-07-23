# Reachability

Analyzes how controllable a system is by reading the structure of the controllability
Gramian `W = Co Co^T`. Its eigenvectors are the principal reach directions and its
eigenvalues say how cheaply the endpoint moves in each.

## Methods

```python
W = engine.reachability.gramian(control)
vals, vecs = engine.reachability.eig(control)          # eigenvalues (strong to weak), eigenvectors
energy, dirs = engine.reachability.energy_per_direction(control)  # min control energy per direction
axes, dirs = engine.reachability.ellipsoid(control)    # reach ellipsoid semi-axes
cond = engine.reachability.condition_number(control)   # anisotropy of reachability
ms = engine.reachability.measures(control)             # scalar measures dict
report = engine.reachability.summary(control)          # full structured report
engine.reachability.print_summary(control, name="my system")
```

## Interpreting the metrics

- **eigenvalues** --- a large eigenvalue is a cheaply reachable direction, a small one is
  weakly controllable (expensive).
- **energy per direction** --- `1 / eigenvalue`, the minimum control effort to move the
  endpoint a unit in that principal direction.
- **condition number** --- ratio of strongest to weakest reach; large means highly
  anisotropic and fold-prone.
- **log det** --- the log volume of the reachable set, a single-number reachability score.
- **weakest / strongest direction** --- the eigenvectors telling you where control
  authority is scarce or abundant.
