# ============================================================================
# grace -- Gramian-based Reachability Analysis and Control Engine:
# ============================================================================
# A lightweight framework for nonlinear optimal control by controllability
# Gramian shooting on CasADi-compiled analytic derivatives.  Provide a dynamics
# function, build a system, and the engine gives control solutions to any state.
#
#   import grace
#   system = grace.build(dynamics, nx, nu, N, z0, dt)
#   engine = grace.GRACE(system)
#   control = engine.shooting.lambda_shoot(target)
# ============================================================================

from .core import build, build_cached, System
from .engine import GRACE
from .codesign import Codesign

__version__ = "0.1.0"

__all__ = ["build", "build_cached", "System", "GRACE", "Codesign"]
