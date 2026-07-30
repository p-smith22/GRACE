# Import packages:
import numpy as np
import casadi as ca
from grace import Codesign

# Verify codesign and return pass/fail records:
def run():

    # Collect results:
    results = []

    # Parameterized double integrator: control authority scales with the design param:
    def dynamics(x, u, p):
        return ca.vertcat(x[2], x[3], p * u[0], p * u[1])

    # Design objective grows with the parameter:
    def objective(p):
        return p ** 2

    # Run codesign over a weight sweep:
    cd = Codesign(dynamics, 4, 2, 60, [0, 0, 0, 0.0], 0.1)
    control, param, front = cd.optimize(np.array([8.0, 0, 0, 0]), "g", objective,
                                        p0=1.0, p_bounds=(0.5, 2.0),
                                        weights=np.linspace(0, 20, 9),
                                        job="test_codesign", plot=False)

    # The front spreads across the parameter range:
    params = [f["param"] for f in front]
    spread = max(params) - min(params)
    results.append(("codesign front spreads", spread > 0.5,
                    f"param range {min(params):.2f}..{max(params):.2f}"))

    # The selected design is within bounds:
    results.append(("codesign param in bounds", 0.5 - 1e-6 <= param <= 2.0 + 1e-6,
                    f"selected param {param:.3f}"))

    # Return the records:
    return results

# Run standalone:
if __name__ == "__main__":
    for name, ok, detail in run():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
