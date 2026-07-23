# ============================================================================
# tracking -- closed-loop trajectory tracking:
# ============================================================================
# Builds the feedback gains that keep the presolved trajectory on course under
# disturbance.  The simulation itself lives in utils.simulate.
# ============================================================================

from .lqr import lqr_gains as _lqr_gains


class Tracking:

    # Bind the tracking tools to a system:
    def __init__(self, system):
        self.system = system

    # Design trajectory-tracking LQR gains for a control tape:
    def lqr_gains(self, control, Q, R):
        return _lqr_gains(self.system, control, Q, R)


__all__ = ["Tracking"]
