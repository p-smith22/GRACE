# ============================================================================
# test_optimizer.py -- verify the reference optimizer:
# ============================================================================
# The direct optimizer must reach the target and match the shoot's cost on a
# simple problem, since both minimize the same effort objective.
# ============================================================================

import numpy as np

import grace
from tests.systems import double_integrator


# Verify the optimizer and return pass/fail records:
def run():

    # Collect results:
    results = []
    D = grace.GRACE(double_integrator())
    target = np.array([8.0, 0, 0, 0])

    # Optimizer reaches the target:
    Uo = D.optimizer.optimize(target)
    err = np.linalg.norm(D.system.endpoint(Uo) - target)
    results.append(("optimizer reaches target", err < 1e-4, f"endpoint err {err:.2e}"))

    # Optimizer cost matches the shoot within a small gap:
    Us = D.shooting.lambda_shoot(target)
    gap = abs(float(Uo @ Uo) - float(Us @ Us)) / float(Us @ Us)
    results.append(("optimizer cost matches shoot", gap < 0.05,
                    f"shoot {float(Us @ Us):.1f} vs opt {float(Uo @ Uo):.1f} (gap {gap * 100:.1f}%)"))

    # Return the records:
    return results


# Run standalone:
if __name__ == "__main__":
    for name, ok, detail in run():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
