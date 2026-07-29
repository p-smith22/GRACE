# Import SLSQP solver::
from .slsqp import optimize as _optimize

# Optimizer class:
class Optimizer:

    # Bind the optimizer to a system:
    def __init__(self, system):
        self.system = system

    # Optimize the minimum-effort control to a target:
    def optimize(self, target, obstacles=None, R=None, pos_idx=(0, 1),
                 U0=None, maxit=500):
        return _optimize(self.system, target, obstacles=obstacles, R=R,
                         pos_idx=pos_idx, U0=U0, maxit=maxit)

# Name:
__all__ = ["Optimizer"]
