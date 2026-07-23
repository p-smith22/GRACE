# ============================================================================
# test_reachability.py -- verify the reachability analysis:
# ============================================================================
# The Gramian must be symmetric, its eigenvalues sorted and nonnegative, and
# the derived measures self-consistent.
# ============================================================================

import numpy as np

import grace
from tests.systems import double_integrator


# Verify reachability metrics and return pass/fail records:
def run():

    # Collect results:
    results = []
    D = grace.GRACE(double_integrator())
    U = D.shooting.lambda_shoot(np.array([8.0, 0, 0, 0]))

    # Gramian is symmetric:
    W = D.reachability.gramian(U)
    sym = np.linalg.norm(W - W.T)
    results.append(("gramian symmetric", sym < 1e-8, f"||W - W^T|| {sym:.2e}"))

    # Eigenvalues are nonnegative and sorted descending:
    vals, vecs = D.reachability.eig(U)
    sorted_ok = np.all(np.diff(vals) <= 1e-9) and vals[-1] > -1e-9
    results.append(("eigenvalues sorted nonneg", sorted_ok, f"min {vals[-1]:.2e} max {vals[0]:.2e}"))

    # Energy per direction is the reciprocal eigenvalue:
    energy, _ = D.reachability.energy_per_direction(U)
    consistent = np.allclose(energy, 1.0 / vals, rtol=1e-4)
    results.append(("energy = 1/eigenvalue", consistent, "reciprocal check"))

    # Return the records:
    return results


# Run standalone:
if __name__ == "__main__":
    for name, ok, detail in run():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
