# Import packages:
import numpy as np
import grace
from tests.systems import cart_pole

# Verify Newton reaches a reachable target and return pass/fail records:
def run():

    # Collect results:
    results = []

    # Build a cart-pole engine and pick a reachable target:
    E = grace.GRACE(cart_pole())
    zt = np.array([0.5, 0, np.pi, 0])

    # Shoot and check the endpoint:
    U = E.shooting.newton_shoot(zt)
    err = np.linalg.norm(E.system.endpoint(U) - zt)
    results.append(("newton_shoot reaches target", err < 1e-6, f"endpoint err {err:.2e}"))

    # Return the records:
    return results

# Run standalone:
if __name__ == "__main__":
    for name, ok, detail in run():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
