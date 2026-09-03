# ============================================================================
# example_surfaces.py -- reachable surfaces across systems, through GRACE
# ============================================================================
# Every tool lives in grace.reachability.analysis. This file only defines
# plants, costs and budgets, and draws the result.
#
# The surface is the sublevel set of the value field V(x), the minimum effort
# lambda_shoot needs to place the endpoint at x. Nothing is linearised, so
# holes, bands and disconnected pieces survive, and one field gives every
# budget in the ladder at once.
#
#   filled shades   sublevel sets of V, one band per cost level
#   black outline   Monte-Carlo truth, sampled on the effort set
#   red dashed      the reach ellipse, which is what a quadratic solve gives
#
# reach applies one Jacobian at the nominal, so it returns a linear image of a
# convex set and is convex whatever the plant does. Every non-elliptic shape
# here comes from solving the true minimum effort problem to each target.
#
# NOTE ON THE MONTE-CARLO REFERENCE: it is only trustworthy when the effort
# space is small. At N_STEP = 12 the cloud concentrates and the support is
# under-reported -- ten times the samples grew the measured area by 22 percent
# on the unicycle. Keep N_STEP low when the numbers are meant to be a check
# rather than a picture.
# ============================================================================
import os
import time
import hashlib
import numpy as np
import casadi as ca
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grace
from grace.reachability import analysis as ra

NG = 320
CACHE = "./_cache"
JIT = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"]}}
N_STEP = 8
SUBSTEPS = 24
# Solve the grid uniformly. Contour-targeted refinement is cheaper per unit of
# resolution, but it fills the cells it does not solve from their nearest
# neighbour, and those fills show up as streaks through the surface. A capped
# iteration budget does the same thing for a different reason: it truncates
# cells that would have converged. Both are false economies here:
NG_COARSE = 56
REFINE = 0
MAX_IT = None

# The inner loop runs to this step tolerance. V is stationary at the optimum,
# so loosening it perturbs the value only to second order while cutting the
# iteration count roughly in proportion to log(1/ftol). Set to None to use the
# solver default of 1e-6, and use ra.ftol_check to measure what it costs:
FTOL = 1e-3
BUDGET_FRAC = [0.2, 0.4, 0.6, 0.8, 1.0]

# === PLANTS ===
# CasADi dynamics, the build arguments, and the two reported coordinates:
def integrator():

    def f(z, u):
        return ca.vertcat(z[1], u[0])

    return dict(name="double integrator", f=f, nx=2, nu=1, T=3.0, dims=(0, 1),
                labels=("endpoint position", "endpoint rate"))

def unicycle():

    def f(z, u):
        return ca.vertcat(ca.cos(z[2]), ca.sin(z[2]), u[0])

    return dict(name="unicycle", f=f, nx=3, nu=1, T=3.0, dims=(0, 1),
                labels=("endpoint x", "endpoint y"))

def duffing():

    def f(z, u):
        return ca.vertcat(z[1], z[0] - z[0] ** 3 - 0.1 * z[1] + u[0])

    return dict(name="Duffing double well", f=f, nx=2, nu=1, T=4.0,
                dims=(0, 1), labels=("endpoint position", "endpoint rate"))

def van_der_pol():

    def f(z, u):
        return ca.vertcat(z[1], (1 - z[0] ** 2) * z[1] - z[0] + u[0])

    return dict(name="Van der Pol", f=f, nx=2, nu=1, T=8.0, dims=(0, 1),
                labels=("endpoint position", "endpoint rate"))

def pendulum():

    def f(z, u):
        return ca.vertcat(z[1], -ca.sin(z[0]) + u[0])

    return dict(name="pendulum", f=f, nx=2, nu=1, T=10.0, dims=(0, 1),
                labels=("endpoint angle", "endpoint rate"))

# === EFFORT COST ===
# Propellant, smoothed. Cost is close to total impulse, so spending everything
# on a few steps is barely dearer than spreading it and the effort set gains
# corners. True L1 has a sign function for its gradient, which has no inverse,
# so lambda_shoot could not use it; the smoothing is what makes dinv exist.
# Returned as (f, dinv, dprime): grad f = u / sqrt(u^2 + e^2) = s, which
# inverts to u = e s / sqrt(1 - s^2) on |s| < 1:
def fuel_cost(e=0.05):

    def f(U):
        U = np.asarray(U, float)
        return float(np.sum(np.sqrt(U ** 2 + e ** 2) - e))

    def dinv(s):
        s = np.clip(np.asarray(s, float), -1.0 + 1e-9, 1.0 - 1e-9)
        return e * s / np.sqrt(1.0 - s ** 2)

    def dprime(s):
        s = np.clip(np.asarray(s, float), -1.0 + 1e-9, 1.0 - 1e-9)
        return e / (1.0 - s ** 2) ** 1.5

    return (f, dinv, dprime)

# === BUILD CACHE ===
# The same CasADi dynamics that go into grace.build are rolled separately for
# the Monte-Carlo reference, so the check does not route through the thing
# being checked. Cached on the build arguments, so a changed horizon or step
# count invalidates the entry rather than silently reusing a stale graph:
def cached_endpoint(plant, N, dt, substeps=SUBSTEPS, chunk=4000):

    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{plant['name']}|{plant['nx']}|{plant['nu']}|{N}|"
                      f"{dt}|{substeps}|{plant['dims']}"
                      .encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"endpoint_{key}.casadi")

    if os.path.exists(path):
        F = ca.Function.load(path)
    else:
        # The control is held across the step, but the step is integrated in
        # substeps -- the SAME substeps grace.build_cached is given. A single
        # RK4 step per control is unstable on a stiff plant at these horizons,
        # and if the reference integrates differently from the framework the
        # comparison is between two different ODEs rather than two methods:
        a = ca.MX.sym("a", N * plant["nu"])
        z = ca.DM.zeros(plant["nx"])
        h = dt / substeps
        for k in range(N):
            u = a[k * plant["nu"]:(k + 1) * plant["nu"]]
            for _ in range(substeps):
                k1 = plant["f"](z, u)
                k2 = plant["f"](z + 0.5 * h * k1, u)
                k3 = plant["f"](z + 0.5 * h * k2, u)
                k4 = plant["f"](z + h * k3, u)
                z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        e = ca.vertcat(z[plant["dims"][0]], z[plant["dims"][1]])
        F = ca.Function("endpoint", [a], [e], JIT)
        F.save(path)

    Fm = F.map(chunk)

    def endpoint(A):
        A = np.atleast_2d(A)
        out = np.empty((len(A), 2))
        for i in range(0, len(A), chunk):
            B = A[i:i + chunk]
            if len(B) == chunk:
                out[i:i + chunk] = np.array(Fm(B.T)).T
            else:
                out[i:i + len(B)] = np.array(F(B.T)).T.reshape(len(B), 2)
        return out

    return endpoint

# === MONTE-CARLO REFERENCE ===
def truth(endpoint, cost_np, c, nd, n, rng):

    # Bound the effort set along random rays, then rejection sample it:
    W = rng.standard_normal((200, nd))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    lo = np.zeros(200)
    hi = np.ones(200)
    for _ in range(50):
        hi = np.where(cost_np(hi[:, None] * W) < c, hi * 2.0, hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        big = cost_np(mid[:, None] * W) > c
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    R = 1.15 * float(hi.max())

    got = []
    have = 0
    while have < n:
        A = rng.uniform(-R, R, (200000, nd))
        A = A[cost_np(A) <= c]
        if len(A):
            got.append(endpoint(A))
            have += len(A)
    Q = np.vstack(got)[:n]

    return Q[np.all(np.isfinite(Q), axis=1)]

# A boundary curve has to be filled as a polygon; rasterising its vertices as
# scattered points leaves gaps that fill_holes can never close:
def fill_poly(V, ext):

    from skimage.draw import polygon as skpoly

    M = np.zeros((NG, NG), bool)
    r, c = skpoly((V[:, 0] - ext[0]) / (ext[1] - ext[0]) * NG,
                  (V[:, 1] - ext[2]) / (ext[3] - ext[2]) * NG, shape=(NG, NG))
    M[r, c] = True

    return M

def rast(P, ext, close=2):

    ix = ((P[:, 0] - ext[0]) / (ext[1] - ext[0]) * NG).astype(int)
    iy = ((P[:, 1] - ext[2]) / (ext[3] - ext[2]) * NG).astype(int)
    ok = (ix >= 0) & (ix < NG) & (iy >= 0) & (iy < NG)
    M = np.zeros((NG, NG), bool)
    M[ix[ok], iy[ok]] = True
    M = ndimage.binary_closing(M, np.ones((3, 3)), iterations=close)

    return ndimage.binary_fill_holes(M)

def score(M, GT):

    inter = (M & GT).sum()

    return (inter / max((M | GT).sum(), 1), inter / GT.sum(),
            (M & ~GT).sum() / max(M.sum(), 1))

# === CASES ===
# Plant, GRACE cost tuple (None is the quadratic default), the same cost
# vectorised for the sampler, and the budget:
CASES = [(integrator, None, lambda A: 0.5 * np.sum(A ** 2, 1), 6.0),
         (unicycle, None, lambda A: 0.5 * np.sum(A ** 2, 1), 8.0),
         (unicycle, fuel_cost(), lambda A: np.sum(np.abs(A), 1), 6.0),
         (duffing, None, lambda A: 0.5 * np.sum(A ** 2, 1), 2.0),
         (van_der_pol, None, lambda A: 0.5 * np.sum(A ** 2, 1), 2.0),
         (pendulum, None, lambda A: 0.5 * np.sum(A ** 2, 1), 4.0)]

if __name__ == "__main__":

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.5))
    axes = axes.ravel()

    print(f"{'case':30s} {'solves':>7s} {'fail':>6s} {'grid':>5s} {'IoU':>6s} "
          f"{'cover':>6s} {'false':>6s} {'ellipse':>8s} {'t(s)':>7s}")

    for ax, (make, cost, cost_np, c) in zip(axes, CASES):

        plant = make()
        dt = plant["T"] / N_STEP
        nd = N_STEP * plant["nu"]

        # Framework path. target_idx makes the endpoint exactly the two
        # plotted states, so every other state is free rather than pinned --
        # asking a unicycle for a position AND a heading is a far harder
        # problem than the one being drawn, and usually has no solution.
        # substeps keeps the integration accurate without adding controls,
        # and job= compiles the graph once and reloads it thereafter:
        system = grace.build_cached(plant["f"], plant["nx"], plant["nu"],
                                    N_STEP, np.zeros(plant["nx"]), dt,
                                    job=plant["name"].replace(" ", "_"),
                                    target_idx=list(plant["dims"]),
                                    substeps=SUBSTEPS)
        engine = grace.GRACE(system)
        U0 = np.zeros(nd)

        e0 = np.asarray(system.endpoint(U0)).ravel()

        # Reference path: the same dynamics, rolled and cached separately:
        endpoint = cached_endpoint(plant, N_STEP, dt, SUBSTEPS)
        P = truth(endpoint, cost_np, c, nd, 200000, rng)
        # Extent from percentiles, not from min and max. A stiff plant can
        # send an explicit integrator into numerical blowup, and those samples
        # come back large but finite, so a min/max extent hands the whole grid
        # to a handful of divergent trajectories. It must also surround the
        # nominal endpoint, since an unstable plant puts the cloud well away
        # from where the nominal lands and a seed on the edge of the extent
        # has almost no neighbours to expand into:
        q_lo = np.percentile(P, 0.1, axis=0)
        q_hi = np.percentile(P, 99.9, axis=0)
        pad = 0.05 * np.maximum(q_hi - q_lo, 1e-9)
        lo = np.minimum(q_lo - pad, e0 - pad)
        hi = np.maximum(q_hi + pad, e0 + pad)
        ext = [lo[0], hi[0], lo[1], hi[1]]

        # Samples outside that window are the blowups, and they would
        # otherwise be rasterised into the reference as reachable:
        P = P[(P[:, 0] >= ext[0]) & (P[:, 0] <= ext[1])
              & (P[:, 1] >= ext[2]) & (P[:, 1] <= ext[3])]
        GT = rast(P, ext)

        # One value field serves every budget in the ladder:
        budgets = [c * fr for fr in BUDGET_FRAC]
        t0 = time.time()
        V, xs, ys, ctrl, n1, f1 = ra.value_field(engine, system, U0, ext,
                                                 NG_COARSE, budgets[-1],
                                                 cost=cost, dims=(0, 1),
                                                 max_it=MAX_IT, ftol=FTOL)
        V, n2 = ra.repair(engine, system, U0, V, xs, ys, ctrl, cost=cost,
                          dims=(0, 1), ftol=FTOL)

        # Double the resolution only where a contour passes, which is far
        # cheaper than a uniform sweep at the same density:
        f2 = 0
        for _ in range(REFINE):
            V, xs, ys, ctrl, n3, f3 = ra.refine_field(engine, system, V, xs,
                                                      ys, ctrl, budgets,
                                                      cost=cost,
                                                      max_it=MAX_IT)
            n2 += n3
            f2 += f3
        hz = ra.hybrid_zonotope(V, xs, ys, budgets)
        dt_build = time.time() - t0

        masks = [ra.sublevel(V, xs, ys, ext, b, NG) for b in budgets]
        s = score(masks[-1], GT)

        # The quadratic answer, from the existing reach. A cost without an
        # invertible marginal map cannot form the costate trace at all, so the
        # ellipse for those cases is the quadratic surrogate:
        rep = engine.reachability.reach(
            U0, budgets[-1],
            cost=cost if ra.is_full_cost(cost) else None,
            dims=(0, 1))
        EL = np.asarray(rep["ellipse"]) + e0
        ELM = fill_poly(EL, ext)
        se = score(ELM, GT)

        tag = "quadratic" if cost is None else "propellant"
        name = f"{plant['name']}, {tag}"
        print(f"{name:30s} {n1 + n2:7d} {f1 + f2:6d} {len(xs):5d} {s[0]:6.3f} "
              f"{s[1]:6.3f} {s[2]:6.3f} {se[0]:8.3f} {dt_build:7.1f}")

        shades = plt.cm.viridis(np.linspace(0.88, 0.18, len(masks)))
        for M, col in zip(masks[::-1], shades):
            over = np.zeros(M.shape + (4,))
            over[M] = col
            ax.imshow(np.transpose(over, (1, 0, 2)), origin="lower", extent=ext)

        gx = np.linspace(ext[0], ext[1], NG)
        gy = np.linspace(ext[2], ext[3], NG)
        ax.contour(gx, gy, GT.T, [0.5], colors="k", linewidths=1.7)
        ax.plot(EL[:, 0], EL[:, 1], color="crimson", lw=1.4, ls="--")

        ax.set_title(f"{name}\nsurface IoU {s[0]:.3f}   ellipse IoU "
                     f"{se[0]:.3f}", fontsize=9)
        ax.set_xlabel(plant["labels"][0], fontsize=8)
        ax.set_ylabel(plant["labels"][1], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])

    fig.suptitle("Reachable surfaces through GRACE: shading is the value field "
                 "by cost level, black is Monte-Carlo truth, red dashed is the "
                 "reach ellipse", fontsize=12)
    fig.tight_layout()
    fig.savefig("surfaces.png", dpi=110, bbox_inches="tight")
    print("\nfigure written to surfaces.png")