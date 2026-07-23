# Shooting

The shooting group generates control solutions. `lambda_shoot` is multi-use: passing
obstacles switches it from a simple minimum-effort solve to hard-constrained obstacle
avoidance.

## newton_shoot

Reaches a feasible endpoint by Gauss-Newton least-norm steps. Feasible but not optimal.

```python
control = engine.shooting.newton_shoot(target, U0=None, it=25)
```

## lambda_shoot (simple)

Minimum-effort optimal control to a target: projected-gradient descent on control cost
confined to the feasible manifold `g(U) = target`.

```python
control = engine.shooting.lambda_shoot(target)
```

## lambda_shoot (obstacle avoidance)

Passing `obstacles` switches to the hard-constrained solve: a coarse A* seeds the routing,
a feasibility phase drives the trajectory clear of every obstacle without ever breaching
one, and a predictor-corrector refine lowers control cost while staying feasible.

```python
control = engine.shooting.lambda_shoot(target, obstacles=obstacles, R=radius, pos_idx=(0, 1))
```

- `obstacles` --- list of 2D obstacle centers
- `R` --- safe radius each obstacle must be cleared by
- `pos_idx` --- which two state components are position (must match `build`)

Obstacle avoidance requires `pos_idx` to have been set in `grace.build`, so a position
Jacobian exists.
