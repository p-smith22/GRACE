# Optimizer

A reference direct optimizer built on the same compiled rollout the shooting solvers use.
It minimizes control effort subject to the endpoint constraint (and obstacle clearance if
obstacles are passed) with SLSQP. Because the endpoint map and its Jacobian are compiled,
each evaluation is cheap, so this is an efficient benchmark to compare the shoot against.

## optimize

```python
control = engine.optimizer.optimize(target, obstacles=None, R=None, pos_idx=(0, 1),
                                    U0=None, maxit=500)
```

- multi-use like `lambda_shoot`: passing `obstacles` adds clearance inequalities
- `U0` --- optional warm start (for example, the shoot's solution)

## Comparing shoot vs optimizer

`engine.utils.compare` runs both on the same problem, times them, and reports the four
comparison metrics. See utils.md.
