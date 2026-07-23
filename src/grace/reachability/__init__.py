# ============================================================================
# reachability -- controllability and reachability analysis:
# ============================================================================
# Extracts the structure of the controllability Gramian so a system can be
# benchmarked for how controllable it is and where its weak directions lie.
# ============================================================================

from . import analysis


class Reachability:

    # Bind the reachability tools to a system:
    def __init__(self, system):
        self.system = system

    # Controllability Gramian at a control tape:
    def gramian(self, control, reg=0.0):
        return analysis.gramian(self.system, control, reg=reg)

    # Eigenvalues and eigenvectors of the Gramian (strong to weak):
    def eig(self, control):
        return analysis.eig(self.system, control)

    # Minimum control energy per principal reach direction:
    def energy_per_direction(self, control):
        return analysis.energy_per_direction(self.system, control)

    # Controllability ellipsoid semi-axes and orientation:
    def ellipsoid(self, control):
        return analysis.ellipsoid(self.system, control)

    # Condition number of the Gramian:
    def condition_number(self, control):
        return analysis.condition_number(self.system, control)

    # Scalar controllability measures:
    def measures(self, control):
        return analysis.measures(self.system, control)

    # Full controllability report:
    def summary(self, control):
        return analysis.summary(self.system, control)

    # Print the controllability report:
    def print_summary(self, control, name="system"):
        return analysis.print_summary(self.system, control, name=name)


__all__ = ["Reachability"]
