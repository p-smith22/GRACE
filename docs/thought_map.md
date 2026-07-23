# GRACE --- thought map for what's next

The package is complete and verified (15/15 benchmarks pass across shooting, tracking,
reachability, optimizer, and codesign). What follows is the roadmap, grouped by your
outstanding list first, then broader directions.

## Your outstanding list

### 1. Control bounds
Add box constraints on the controls (`u_min <= u <= u_max`).
- **Shooting**: clip the trial step inside the line search, and project the reduced-gradient
  step onto the feasible box each iteration. The verified obstacle solver already backtracks;
  bounds slot into the same line-search guard.
- **Optimizer**: pass `bounds` straight to SLSQP --- trivial, and gives a bounded reference to
  compare the bounded shoot against.
- **Design question**: bounds break the clean least-norm structure of `lambda_shoot`. The
  honest approach is an active-set treatment of the bound constraints (like the obstacle
  active set), so the shoot stays a projected-gradient method rather than becoming a generic QP.

### 2. Partial observation (C not identity)
You have the documentation; the machinery is a natural fit.
- The endpoint map becomes `y = C x` rather than the full state. `target_idx` already handles
  axis-aligned selection; a general `C` means the endpoint map is `C @ Zmat[-1]`, and its
  Jacobian is `C @ dg/dU`. This is a small change in `build`.
- **Observation Gramian**: with `C`, the reachability analysis reports controllability of the
  *observed* subspace, which is exactly the "observation gap" analysis you wanted.
- **Koopman extension** (longer horizon): lift the state through observable functions and build
  the endpoint map in the lifted space --- GRACE's structure carries over unchanged since it
  only needs a differentiable rollout.

### 3. Control weighting via R
Weight controls against each other in the cost `int u^T R u dt`.
- The cost gradient `2U` becomes `2 R_block U` where `R_block` is block-diagonal across the
  horizon. Every shoot already works with this gradient abstractly, so it is a one-line change
  to the reduced-gradient computation in `lambda_simple` and `lambda_obstacle`.
- **Stationarity** must use the same weighted gradient to stay consistent --- update
  `diagnostics.stationarity` in lockstep.

## Broader directions

### Wide-system validation
The payoff of "just provide the dynamics." Build a validation matrix over diverse systems
(quadrotor, Dubins car, pendulum-on-cart, planar arm, unicycle, 6DOF aircraft) and confirm
each solves from a cold start. This is the strongest evidence GRACE is general.

### Aircraft as a real benchmark
The 6DOF aircraft dynamics are on disk. Wire them through `build` as a benchmark system so the
suite tests a realistic nonlinear model, not just cart-pole and double-integrator.

### Codesign hardening
The envelope-theorem gradient works, but the front can rail to a bound when the objective is
monotone in the parameter (seen in testing). Add: (a) a check that flags a degenerate front,
(b) a multi-parameter design vector, (c) a proper Pareto-dominance filter rather than a fixed
weight sweep.

### Speed
Obstacle avoidance rebuilds the position Jacobian each iteration. For linear systems this is
constant --- cache it. For nonlinear systems, a low-rank active-set factorization update would
cut the per-iteration solve as the horizon grows.

### Robust obstacle safety (open question)
Presolved obstacle trajectories are open-loop; a disturbance can push the vehicle into an
obstacle. A CBF safety filter layered on the presolve would give a hard clearance guarantee
under bounded disturbance --- the right closed-loop form for obstacles (plain LQR tracking is
not, since it minimizes deviation rather than preserving clearance).

## Suggested order

1. Control weighting via R (smallest, unblocks weighted stationarity everywhere)
2. Control bounds (active-set, reuses obstacle machinery)
3. Partial observation / C matrix (small build change, big analysis payoff)
4. Aircraft benchmark + wide-system validation matrix
5. Codesign hardening
6. Robust obstacle safety (CBF) --- the deepest, most research-flavored piece
