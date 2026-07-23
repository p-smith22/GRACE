# Caching compiled graphs

Compiling the CasADi graph is the slow part of building a system. Give a system a `job`
name and its compiled graph is saved to the data directory, so later runs reload it
instead of recompiling.

## Build with caching

```python
from grace.core import build_cached

# First run compiles and saves under data/cart_pole/; later runs reload:
system = build_cached(dynamics, nx, nu, N, z0, dt, job="cart_pole",
                     pos_idx=(0, 1), rebuild=False)
```

- `job` --- names the cache subdirectory under the data directory
- `rebuild=True` --- force recompilation even if a cache exists
- `data_dir` --- override the data directory (defaults to `data`)

## Manual save and load

```python
from grace.core import build, save, load, has_cache

system = build(dynamics, nx, nu, N, z0, dt, pos_idx=(0, 1), job="cart_pole")
save(system)                      # writes data/cart_pole/
if has_cache("cart_pole"):
    system = load("cart_pole")    # reloads the compiled graph
```

The position Jacobian used for obstacle avoidance is rebuilt automatically on reload, so a
cached system supports obstacle avoidance without recompiling from the dynamics.
