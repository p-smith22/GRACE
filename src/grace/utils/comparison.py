# Import packages;
import time
from .diagnostics import diagnostics, clearance

# Print a standardized one-line summary of a solve:
def print_result(method, mode, cost, stationarity, endpoint_error, time_ms,
                 clearance_val=None):

    # Assemble the standardized fields:
    line = (f"[{method:>12}] mode={mode:<8} "
            f"cost={cost:9.2f}  stat={stationarity:.2e}  "
            f"endpt={endpoint_error:.2e}  time={time_ms:7.1f}ms")

    # Append clearance when the run was an obstacle run:
    if clearance_val is not None:
        line += f"  clr={clearance_val:.2f}"

    # Print the standardized line:
    print(line)

# Compare the shoot against the optimizer on one problem:
def compare(engine, target, obstacles=None, R=None, pos_idx=(0, 1),
            verbose=True, plot=None):

    # Designate mode depending on presence of obstacle:
    mode = "obstacle" if obstacles is not None else "simple"
    system = engine.system

    # Time the GRACE shoot:
    t0 = time.perf_counter()
    U = engine.shooting.lambda_shoot(target, obstacles=obstacles, R=R, pos_idx=pos_idx)
    t_shoot = (time.perf_counter() - t0) * 1e3

    # Time the optimizer:
    t0 = time.perf_counter()
    Uo = engine.optimizer.optimize(target, obstacles=obstacles, R=R, pos_idx=pos_idx)
    t_opt = (time.perf_counter() - t0) * 1e3

    # Diagnose both solutions:
    ds = diagnostics(system, U, target)
    do = diagnostics(system, Uo, target)

    # Add clearance for obstacle runs:
    cs = clearance(system, U, obstacles, R, pos_idx) if obstacles is not None else None
    co = clearance(system, Uo, obstacles, R, pos_idx) if obstacles is not None else None

    # Print the standardized lines:
    if verbose:
        print_result("lambda_shoot", mode, ds["cost"], ds["stationarity"],
                     ds["endpoint_error"], t_shoot, cs)
        print_result("optimizer", mode, do["cost"], do["stationarity"],
                     do["endpoint_error"], t_opt, co)

    # Collect the four-metric comparison:
    metrics = {"time_ms": (t_shoot, t_opt),
               "cost": (ds["cost"], do["cost"]),
               "endpoint_err": (ds["endpoint_error"], do["endpoint_error"]),
               "stationarity": (ds["stationarity"], do["stationarity"])}

    # Optionally save the comparison panel:
    if plot is not None:
        from .plotting import compare_panel
        compare_panel(metrics, labels=("GRACE", "optimizer"), save=plot)

    # Return the solutions and metrics:
    return dict(U=U, U_optimizer=Uo, metrics=metrics,
                speedup=t_opt / t_shoot if t_shoot > 0 else float("nan"))
