# Import codesign function:
from .codesign import codesign as _codesign

# Create usage class:
class Codesign:

    # Bind codesign to an engine's problem definition:
    def __init__(self, dynamics, nx, nu, N, z0, dt):
        self.dynamics = dynamics
        self.nx = nx
        self.nu = nu
        self.N = N
        self.z0 = z0
        self.dt = dt

    # Optimize a named design parameter against a design objective:
    def optimize(self, target, param_name, objective, p0, p_bounds,
                 weights=None, substeps=1, figures_dir="figures",
                 job="codesign", plot=True, target_idx=None):
        return _codesign(self.dynamics, self.nx, self.nu, self.N, self.z0,
                         self.dt, target, param_name, objective, p0, p_bounds,
                         weights=weights, substeps=substeps,
                         figures_dir=figures_dir, job=job, plot=plot,
                         target_idx=target_idx)

# Name:
__all__ = ["Codesign"]