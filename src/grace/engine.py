# ============================================================================
# engine.py -- the GRACE engine:
# ============================================================================
# Wraps a compiled system and exposes the solver groups as namespaces:
# engine.shooting, engine.optimizer, engine.tracking, engine.reachability,
# engine.codesign, and engine.utils.  This is the object the README drives.
# ============================================================================

from .shooting import Shooting
from .optimizer import Optimizer
from .tracking import Tracking
from .reachability import Reachability
from .utils import Utils


class GRACE:

    # Attach the solver namespaces to a compiled system:
    def __init__(self, system):

        # Store the compiled system:
        self.system = system

        # Expose the solver groups as namespaces:
        self.shooting = Shooting(system)
        self.optimizer = Optimizer(system)
        self.tracking = Tracking(system)
        self.reachability = Reachability(system)
        self.utils = Utils(self)
