# Import LQR tracking:
from .lqr import lqr_gains as _lqr_gains

# Tracking class:
class Tracking:

    # Bind the tracking tools to a system:
    def __init__(self, system):
        self.system = system

    # LQR gains computation:
    def lqr_gains(self, control, Q, R):
        return _lqr_gains(self.system, control, Q, R)

# Name:
__all__ = ["Tracking"]
