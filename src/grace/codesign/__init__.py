# Import codesign functions:
from .codesign import codesign as _codesign
from .codesign import scan as _scan

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
                 weights=None, substeps=1, save="figures/codesign.png",
                 job="codesign", plot=True, target_idx=None, norm="cheby",
                 n_anchor=9, beta=100.0, rho=1e-3, debug=False,
                 p_tol=1e-5, max_outer=40, jit=True, jit_flags="-O1",
                 cache_dir=".grace_cache", filter_dominated=True):
        return _codesign(self.dynamics, self.nx, self.nu, self.N, self.z0,
                         self.dt, target, param_name, objective, p0, p_bounds,
                         weights=weights, substeps=substeps,
                         save=save, job=job, plot=plot,
                         target_idx=target_idx, norm=norm,
                         n_anchor=n_anchor, beta=beta, rho=rho, debug=debug,
                         p_tol=p_tol, max_outer=max_outer,
                         jit=jit, jit_flags=jit_flags,
                         cache_dir=cache_dir,
                         filter_dominated=filter_dominated)

    # Trace the front by solving directly at each design value:
    def scan(self, target, param_name, objective, p_values, substeps=1,
             save="figures/scan.png", job="scan", plot=True, target_idx=None,
             filter_dominated=True):
        return _scan(self.dynamics, self.nx, self.nu, self.N, self.z0,
                     self.dt, target, param_name, objective, p_values,
                     substeps=substeps, save=save, job=job, plot=plot,
                     target_idx=target_idx, filter_dominated=filter_dominated)

# Name:
__all__ = ["Codesign"]