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
    def lambda_shoot(self, target, constraints=(), U0=None, max_it=None,
                     R_weights=None, **kwargs):

        # A constraint is any expression g(z, u) <= 0.  Nothing classifies it as
        # an obstacle, a state limit or a control bound, because nothing needs
        # to: each one reports which states and controls it touches and only
        # those are differentiated.
        #
        #   lambda z, u: R**2 - ((z[0] - ox)**2 + (z[1] - oy)**2)   keep-out
        #   lambda z, u: z[4] - 0.35                                attitude
        #   lambda z, u: u[0] - 11.0                                thrust
        #
        if callable(constraints):
            cons = [constraints]
        else:
            cons = list(constraints)

        # Only override the solver's own iteration budget when one was asked
        # for, since a fixed default here is far below what a long horizon needs:
        if max_it is not None:
            kwargs["max_it"] = max_it

        # Run the shoot:
        return _lambda_shoot(self.system, target, constraints=cons, U0=U0,
                             R_weights=R_weights, **kwargs)

# Name:
__all__ = ["Shooting"]