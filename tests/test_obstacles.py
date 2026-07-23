# ============================================================================
# test_obstacles.py -- verify obstacle-avoidance shooting:
# ============================================================================
# The obstacle shoot must clear multiple obstacles at different locations by
# the shared safe radius while reaching the target.
# ============================================================================

import numpy as np

import grace
from tests.systems import double_integrator


# Verify multi-obstacle avoidance and return pass/fail records:
def run():

    # Collect results:
    results = []
    D = grace.GRACE(double_integrator())

    # Multiple obstacles at different places, one shared radius:
    OBS = [np.array([3.5, 1.8]), np.array([7.0, -1.8]), np.array([10.5, 1.8])]
    R = 2.4
    target = np.array([14.0, 0, 0, 0])
    U = D.shooting.lambda_shoot(target, obstacles=OBS, R=R, pos_idx=(0, 1))

    # Check clearance to every obstacle and the endpoint:
    p = D.system.rollout(U)[:, :2]
    clears = [np.sqrt(np.sum((p - o) ** 2, axis=1)).min() for o in OBS]
    ee = np.linalg.norm(D.system.endpoint(U) - target)
    all_clear = all(c >= R - 1e-2 for c in clears)
    results.append(("multi-obstacle all cleared", all_clear,
                    f"clearances {[float(round(c, 2)) for c in clears]} (need {R})"))
    results.append(("obstacle_shoot reaches target", ee < 1e-2, f"endpoint err {ee:.1e}"))

    # Return the records:
    return results


# Run standalone:
if __name__ == "__main__":
    for name, ok, detail in run():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
