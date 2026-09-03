# Import analysis functions:
from . import analysis


# Reachability class:
class Reachability:

    # Bind the reachability tools to a system:
    def __init__(self, system):
        self.system = system

    # Controllability Gramian at a given control:
    def gramian(self, control, reg=0.0, cost=None, lam=None):
        return analysis.gramian(
            self.system, control, reg=reg, cost=cost, lam=lam
        )

    # Eigenvalues and eigenvectors of the Gramian (strongest to weakest):
    def eig(self, control, cost=None, lam=None):
        return analysis.eig(
            self.system, control, cost=cost, lam=lam
        )

    # Minimum control energy per principal reach direction:
    def energy_per_direction(self, control, cost=None, lam=None):
        return analysis.energy_per_direction(
            self.system, control, cost=cost, lam=lam
        )

    # Controllability ellipsoid semi-axes and orientation:
    def ellipsoid(self, control, cost=None, lam=None):
        return analysis.ellipsoid(
            self.system, control, cost=cost, lam=lam
        )

    # Condition number of the Gramian:
    def condition_number(self, control, cost=None, lam=None):
        return analysis.condition_number(
            self.system, control, cost=cost, lam=lam
        )

    # Reachable set on a cost budget, sliced by a coordinate plane:
    def reach(self, control, budget, cost=None, lam=None,
              dims=(0, 1), n=180):
        return analysis.reach(
            self.system, control, budget, cost=cost,
            lam=lam, dims=dims, n=n
        )

    # Hybrid-zonotope approximation of nested reachable sets:
    def hybrid_zonotope(self, control, budgets, cost=None, lam=None,
                        dims=(0, 1), n=180, source="true"):
        return analysis.hybrid_zonotope(
            self.system, control, budgets, cost=cost, lam=lam,
            dims=dims, n=n, source=source
        )

    # Vertices of every hybrid-zonotope cell:
    def hz_vertices(self, hz):
        return analysis.hz_vertices(hz)

    # Test whether points lie inside the hybrid-zonotope union:
    def hz_contains(self, hz, points, tol=1e-9):
        return analysis.hz_contains(hz, points, tol=tol)

    # Area of the hybrid-zonotope union:
    def hz_area(self, hz, ng=400):
        return analysis.hz_area(hz, ng=ng)

    # Full controllability report:
    def summary(self, control, cost=None, lam=None):
        return analysis.summary(
            self.system, control, cost=cost, lam=lam
        )

    # Print the controllability report:
    def print_summary(self, control, name="system", cost=None, lam=None):
        return analysis.print_summary(
            self.system, control, name=name, cost=cost, lam=lam
        )


# Public API:
__all__ = ["Reachability"]