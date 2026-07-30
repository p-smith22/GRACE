# Import packages:
import numpy as np
import grace
from tests.systems import double_integrator

# Verify tracking exactness and disturbance rejection, return pass/fail records:
def run():

    # Collect results:
    results = []
    D = grace.GRACE(double_integrator())

    # Solve a nominal and design tracking gains:
    U = D.shooting.lambda_shoot(np.array([8.0, 0, 0, 0]))
    gains, nominal = D.tracking.lqr_gains(U, np.diag([10, 10, 1, 1.0]), np.eye(2) * 0.5)

    # Nominal is exact with feedback and no disturbance:
    Znom, _ = D.utils.simulate(U, gains=gains, nominal=nominal, disturb=None, feedback=True)
    dev = np.linalg.norm(Znom[-1] - nominal[-1])
    results.append(("closed-loop nominal exact", dev < 1e-6, f"nominal dev {dev:.2e}"))

    # Feedback rejects a disturbance:
    rng = np.random.default_rng(1)
    dist = np.zeros((D.system.N, 4))
    g = np.zeros(4)
    for k in range(D.system.N):
        g = 0.9 * g + np.array([0, 0, 0.3, 0.3]) * rng.standard_normal(4)
        dist[k] = g
    Zol, _ = D.utils.simulate(U, gains=gains, nominal=nominal, disturb=dist, feedback=False)
    Zcl, _ = D.utils.simulate(U, gains=gains, nominal=nominal, disturb=dist, feedback=True)
    dol = np.linalg.norm(Zol[-1] - nominal[-1])
    dcl = np.linalg.norm(Zcl[-1] - nominal[-1])
    results.append(("closed-loop rejects disturbance", dcl < dol, f"open {dol:.3f} -> closed {dcl:.3f}"))

    # Return the records:
    return results

# Run standalone:
if __name__ == "__main__":
    for name, ok, detail in run():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
