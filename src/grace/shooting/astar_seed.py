# ============================================================================
# astar_seed.py -- coarse A* routing seed for obstacle avoidance:
# ============================================================================
# Deciding which side of each obstacle to pass is a global routing choice, not
# a local one, so a coarse grid A* makes it.  The resulting path is turned into
# a warm-start control tape by a least-norm shot through waypoints sampled along
# it.  This commits the obstacle shoot to the right homotopy class from a cold
# start, instead of stalling head-on at an obstacle.
# ============================================================================

import numpy as np
import heapq


# Build a warm-start control tape that routes around the obstacles via A*:
def astar_seed(system, zt, OBS, R, pi, res=0.4):

    # Start and goal in the position plane:
    z0p = system.z0[pi]
    ztf = np.asarray(zt, float)
    g = ztf[[system.tidx.index(pi[0]), system.tidx.index(pi[1])]] \
        if (pi[0] in system.tidx and pi[1] in system.tidx) else ztf[:2]

    # A cell is free if it clears every obstacle by a small margin:
    def free(p):
        for o in OBS:
            if np.sum((p - o) ** 2) < (R + 0.15) ** 2:
                return False
        return True

    # Snap between continuous positions and grid nodes:
    def snap(p):
        return (round(p[0] / res), round(p[1] / res))

    def posn(n):
        return np.array([n[0] * res, n[1] * res])

    # Bound the grid to a box around the start, goal, and obstacles:
    xs = [z0p[0], g[0]] + [o[0] for o in OBS]
    ys = [z0p[1], g[1]] + [o[1] for o in OBS]
    xlo, xhi = min(xs) - R - 1, max(xs) + R + 1
    ylo, yhi = min(ys) - R - 1, max(ys) + R + 1

    # Run A* over the eight-connected grid:
    start = snap(z0p)
    goal = snap(g)
    openh = [(0, start)]
    came = {}
    gsc = {start: 0}
    nb = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    found = False
    cnt = 0
    while openh and cnt < 50000:
        cnt += 1
        _, cur = heapq.heappop(openh)

        # Stop when the goal cell is reached:
        if cur == goal:
            found = True
            break

        # Expand the free neighbors within the grid bounds:
        for dx, dy in nb:
            n2 = (cur[0] + dx, cur[1] + dy)
            p = posn(n2)
            if p[0] < xlo or p[0] > xhi or p[1] < ylo or p[1] > yhi:
                continue
            if not free(p):
                continue
            ng = gsc[cur] + np.hypot(dx, dy) * res
            if n2 not in gsc or ng < gsc[n2]:
                gsc[n2] = ng
                came[n2] = cur
                h = np.hypot(goal[0] - n2[0], goal[1] - n2[1]) * res
                heapq.heappush(openh, (ng + h, n2))

    # Fall back to a zero seed if no route was found:
    U0 = np.zeros(system.N * system.nu)
    if not found:
        return U0

    # Reconstruct the grid path:
    path = [goal]
    while path[-1] in came:
        path.append(came[path[-1]])
    path = np.array([posn(n) for n in reversed(path)])

    # Turn the path into a control tape by a least-norm shot through waypoints:
    zref = system.rollout(U0)
    Jp = np.array(system.pos_jac(U0))
    npts = min(len(path), 6)
    idxs = np.linspace(0, len(path) - 1, npts).astype(int)
    rows = []
    res_ = []
    for p_ in idxs[1:-1]:
        k = int(np.clip(p_ / (len(path) - 1) * system.N, 2, system.N - 1))
        rows.append(Jp[k])
        res_.append(path[p_] - zref[k, pi])

    # Add the endpoint constraint so the seed reaches the goal:
    e0, Je = system.endpoint_jac(U0)
    rows.append(Je)
    res_.append(zt - e0)

    # Solve the stacked least-norm system for the seed controls:
    A = np.vstack(rows)
    b = np.concatenate(res_)
    return A.T @ np.linalg.solve(A @ A.T + 1e-8 * np.eye(len(b)), b)
