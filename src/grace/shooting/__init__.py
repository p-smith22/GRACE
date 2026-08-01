# Import shooting files:
from .newton_shoot import newton_shoot as _newton_shoot
from .lambda_shoot import lambda_shoot as _lambda_shoot

# Create shooting class:
class Shooting:

    # Bind the shooting solvers to a system:
    def __init__(self, system):
        self.system = system

    # Reach a feasible endpoint by Newton feasibility steps:
    def newton_shoot(self, target, U0=None, it=25, u_lo=None, u_hi=None):

        # The solver works on the reduced target, so the raw target is mapped
        # onto the constrained states here.  Everything below the shooting
        # package therefore sees only reduced targets, which keeps the
        # convention in one place instead of split across call sites:
        zt = self.system.target(target)
        return _newton_shoot(self.system, zt, U0=U0, it=it, u_lo=u_lo, u_hi=u_hi)

    # Lambda shoot for optimal control:
    def lambda_shoot(self, target, obstacles=None, R=None, pos_idx=(0, 1),
                     U0=None, max_it=None, u_lo=None, u_hi=None, R_weights=None,
                     **kwargs):

        # Obstacles are no longer a separate solver.  They enter the same
        # minimum-effort shoot as augmented-Lagrangian terms in the cost
        # gradient, so with no obstacles the obstacle branch never runs and this
        # is exactly the simple lambda shoot it always was:
        if obstacles is not None and R is None:
            raise ValueError("obstacle avoidance needs a keep-out radius -- pass R "
                             "as a scalar, or as one radius per obstacle")

        # The position Jacobian is compiled for the indices given at build time,
        # so solving with different ones would score the constraint on different
        # states than the Jacobian describes.  That still returns something, but
        # the descent directions are wrong and it fails quietly:
        if obstacles is not None:
            built = getattr(self.system, "pos_idx", None)
            if built is None:
                raise ValueError("obstacle avoidance needs a position Jacobian -- "
                                 "rebuild the system with pos_idx=%s"
                                 % (tuple(pos_idx),))
            if list(built) != list(pos_idx):
                raise ValueError(
                    "pos_idx mismatch: the system's position Jacobian was compiled "
                    "for %s but the solve was called with %s -- pass the same "
                    "pos_idx to build and to the solve."
                    % (tuple(int(i) for i in built), tuple(pos_idx)))

        # Only override the solver's own iteration budget when one was asked for,
        # since the default here used to be far below what a long horizon needs:
        if max_it is not None:
            kwargs["max_it"] = max_it

        # Run the unified shoot:
        return _lambda_shoot(self.system, target, obstacles=obstacles, R=R, U0=U0,
                             u_lo=u_lo, u_hi=u_hi, R_weights=R_weights, **kwargs)

# Name:
__all__ = ["Shooting"]