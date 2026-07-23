# ============================================================================
# shooting -- the shooting solvers:
# ============================================================================
# newton_shoot reaches a feasible endpoint.  lambda_shoot finds the minimum-
# effort control to a target, and is multi-use: passing obstacles switches it
# from the simple solve to the hard-constrained obstacle-avoidance solve.
# ============================================================================

from .newton import newton_shoot as _newton_shoot
from .lambda_simple import lambda_simple
from .lambda_obstacle import lambda_obstacle


class Shooting:

    # Bind the shooting solvers to a system:
    def __init__(self, system):
        self.system = system

    # Reach a feasible endpoint by Newton feasibility steps:
    def newton_shoot(self, target, U0=None, it=25, u_lo=None, u_hi=None):
        return _newton_shoot(self.system, target, U0=U0, it=it, u_lo=u_lo, u_hi=u_hi)

    # Find the minimum-effort control to a target:
    #   without obstacles -- simple projected-gradient shoot,
    #   with obstacles     -- hard-constrained obstacle-avoidance shoot.
    # Optional u_lo/u_hi give per-control box bounds, R_weights weights the cost.
    def lambda_shoot(self, target, obstacles=None, R=None, pos_idx=(0, 1),
                     U0=None, max_it=60, u_lo=None, u_hi=None, R_weights=None):

        # Route to the obstacle solver when obstacles are given:
        if obstacles is not None:
            return lambda_obstacle(self.system, target, obstacles, R,
                                   pos_idx=pos_idx, max_it=max(max_it, 250),
                                   u_lo=u_lo, u_hi=u_hi, R_weights=R_weights)

        # Otherwise run the simple minimum-effort shoot:
        return lambda_simple(self.system, target, U0=U0, max_it=max_it,
                             u_lo=u_lo, u_hi=u_hi, R_weights=R_weights)


__all__ = ["Shooting"]