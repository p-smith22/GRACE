# Import utility functions:
from . import diagnostics as _diagnostics
from . import simulate as _simulate
from . import plotting as _plotting
from . import comparison as _comparison

# Utils class:
class Utils:

    # Bind the utilities to an engine:
    def __init__(self, engine):
        self.engine = engine
        self.system = engine.system

    # Simulate the closed-loop (LQR tracking) or open-loop system:
    def simulate(self, control, gains=None, nominal=None, disturb=None, feedback=True):
        return _simulate.simulate(self.system, control, gains=gains, nominal=nominal,
                                  disturb=disturb, feedback=feedback)

    # Plot the trajectory, states, and controls:
    def plotting(self, trajectory, control, obstacles=None, R=None, pos_idx=(0, 1),
                 state_names=None, control_names=None, title="States and controls",
                 save=None, show_traj=True):
        return _plotting.plotting(self.system, trajectory, control, obstacles=obstacles,
                                  R=R, pos_idx=pos_idx, state_names=state_names,
                                  control_names=control_names, title=title, save=save,
                                  show_traj=show_traj)

    # Overlay nominal, open-loop, and closed-loop trajectories:
    def closed_loop_plot(self, nominal, open_loop, closed, comp_idx=0, comp_name=None,
                         title="Closed-loop tracking", save=None):
        return _plotting.closed_loop(nominal, open_loop, closed, dt=self.system.dt,
                                     comp_idx=comp_idx, comp_name=comp_name,
                                     title=title, save=save)

    # Compute cost, endpoint error, and stationarity of a control:
    def diagnostics(self, control, target, constraints=()):
        return _diagnostics.diagnostics(self.system, control, target, constraints)

    # Compute the stationarity residual of a control:
    def stationarity(self, control, constraints=()):
        return _diagnostics.stationarity(self.system, control, constraints)

    # Compute the minimum obstacle clearance of a control:
    def clearance(self, control, obstacles, R, pos_idx=(0, 1)):
        return _diagnostics.clearance(self.system, control, obstacles, R, pos_idx)

    # Compare the shoot against the optimizer on one problem:
    def compare(self, target, obstacles=None, R=None, pos_idx=(0, 1),
                verbose=True, plot=None):
        return _comparison.compare(self.engine, target, obstacles=obstacles, R=R,
                                   pos_idx=pos_idx, verbose=verbose, plot=plot)

    # Print a standardized result line:
    def print_result(self, method, mode, cost, stationarity, endpoint_error,
                     time_ms, clearance_val=None):
        return _comparison.print_result(method, mode, cost, stationarity,
                                        endpoint_error, time_ms, clearance_val)

# Name:
__all__ = ["Utils"]
