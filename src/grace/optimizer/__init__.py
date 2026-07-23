# ============================================================================
# optimizer -- direct optimization over the compiled system:
# ============================================================================
# An efficient reference solver built on the same compiled rollout the shooting
# solvers use.  Multi-use like lambda_shoot: passing obstacles adds clearance
# constraints.  A warm start is used when provided.
# ============================================================================

from .slsqp import optimize as _optimize


class Optimizer:

    # Bind the optimizer to a system:
    def __init__(self, system):
        self.system = system

    # Optimize the minimum-effort control to a target:
    #   without obstacles -- endpoint-constrained effort minimization,
    #   with obstacles     -- adds obstacle-clearance inequalities.
    def optimize(self, target, obstacles=None, R=None, pos_idx=(0, 1),
                 U0=None, maxit=500):
        return _optimize(self.system, target, obstacles=obstacles, R=R,
                         pos_idx=pos_idx, U0=U0, maxit=maxit)


__all__ = ["Optimizer"]
