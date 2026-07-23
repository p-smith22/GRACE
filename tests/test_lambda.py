# ============================================================================
# test_lambda.py -- verify minimum-effort lambda shooting:
# ============================================================================
# Lambda shoot must be feasible, cost no more than Newton, and its analytic
# Jacobian must match a finite-difference rollout (linearization correctness).
# ============================================================================

import numpy as np

import grace
from tests.systems import cart_pole


# Verify lambda shoot optimality and linearization, return pass/fail records:
def run():

    # Collect results:
    results = []
    E = grace.GRACE(cart_pole())
    zt = np.array([0.5, 0, np.pi, 0])

    # Lambda is feasible:
    Un = E.shooting.newton_shoot(zt)
    Ul = E.shooting.lambda_shoot(zt)
    el = np.linalg.norm(E.system.endpoint(Ul) - zt)
    results.append(("lambda_shoot feasible", el < 1e-2, f"endpoint err {el:.2e}"))

    # Lambda costs no more than Newton:
    results.append(("lambda_shoot cost <= newton",
                    float(Ul @ Ul) <= float(Un @ Un) + 1e-6,
                    f"lambda {float(Ul @ Ul):.1f} vs newton {float(Un @ Un):.1f}"))

    # Analytic Jacobian matches the rollout:
    e0, J = E.system.endpoint_jac(Ul)
    dU = np.random.default_rng(0).standard_normal(E.system.N * E.system.nu) * 1e-4
    err = np.linalg.norm((e0 + J @ dU) - E.system.endpoint(Ul + dU))
    results.append(("analytic Jacobian correct", err < 1e-5, f"||pred-actual|| {err:.2e}"))

    # Return the records:
    return results


# Run standalone:
if __name__ == "__main__":
    for name, ok, detail in run():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
