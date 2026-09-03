# Import package:
import numpy as np
from scipy.optimize import brentq

# === COST CONTRACT ===
# A general cost is the triple (g, dinv, M), all taking and returning the
# stacked control vector of length N*nu:
#
#   g(U)      scalar cost
#   dinv(v)   inverse marginal map, the U solving grad g(U) = v
#   M(v)      curvature at that U, either a flat diagonal of length N*nu for a
#             separable cost or an (N, nu, nu) stack for a coupled one
#
# cost=None selects the quadratic default g(U) = U'U / 2, which is what
# lambda_shoot minimizes implicitly. Every entry is called with a FLAT vector,
# so callers passing U in (N, nu) shape must ravel first.


# A cost is "full" only when it carries the inverse marginal map and the
# curvature as well as the scalar. Some physically real costs cannot: total
# impulse has a sign function for its gradient, so grad g is not invertible and
# no dinv exists. Such a cost is still perfectly usable for the value field,
# which needs the scalar alone, but the ellipse and the costate trace cannot be
# formed from it and fall back to the quadratic surrogate:
def is_full_cost(cost):

    return cost is not None and len(cost) >= 3

# Weighted Gramian at a costate, W = Co M Co':
def _gram_at(system, Co, cost, lam):

    # M is flat (diagonal) for a separable cost, (N, nu, nu) for a coupled one:
    dp = np.asarray(cost[2](Co.T @ lam))
    if dp.ndim == 1:
        return Co @ (dp[:, None] * Co.T)
    CoB = Co.reshape(system.m, -1, system.nu)
    return np.einsum("pki,kij,qkj->pq", CoB, dp, CoB)

# Compute Controllability Gramian:
def gramian(system, U, reg=0.0, cost=None, lam=None):

    # Fetch endpoint Jacobian:
    _, Co = system.endpoint_jac(U)

    # Quadratic weight, the original behaviour:
    if cost is None or len(cost) < 3:
        return Co @ Co.T + reg * np.eye(system.m)

    # A general cost has M depending on the iterate, so it needs the costate:
    if lam is None:
        lam = getattr(system, "_costate", None)
    if lam is None:
        raise ValueError("a general cost needs lam (or system._costate) "
                         "since M depends on the iterate")
    return _gram_at(system, Co, cost, np.asarray(lam).ravel()) \
        + reg * np.eye(system.m)

# Compute eigenvalues and eigenvectors of system:
def eig(system, U, cost=None, lam=None):

    # Fetch Gramian:
    W = gramian(system, U, cost=cost, lam=lam)

    # Compute eigenvalues and eigenvectors:
    vals, vecs = np.linalg.eigh(W)

    # Sort from strongest to weakest:
    order = np.argsort(vals)[::-1]

    # Return eigenvalues and eigenvectors:
    return vals[order], vecs[:, order]

# Minimum control energy to move the endpoint a unit in each principal direction:
def energy_per_direction(system, U, reg=1e-12, cost=None, lam=None):

    # Fetch eigenvalues and eigenvectors:
    vals, vecs = eig(system, U, cost=cost, lam=lam)

    # Exact for a quadratic cost, a local curvature model otherwise:
    return 1.0 / (vals + reg), vecs

# Controllability ellipsoid semi-axes (gives an idea on reachability):
def ellipsoid(system, U, cost=None, lam=None):

    # Compute eigenvalues and eigenvectors:
    vals, vecs = eig(system, U, cost=cost, lam=lam)

    # Return principle axes and their lengths:
    return np.sqrt(np.clip(vals, 0, None)), vecs

# Condition number of the Gramian:
def condition_number(system, U, cost=None, lam=None):

    # Compute eigenvalues:
    vals, _ = eig(system, U, cost=cost, lam=lam)

    # Compute condition number and return:
    return float(vals[0] / vals[-1])

# Endpoint displacement and cost produced by a costate, no solve required:
def _probe(system, Co, cost, lam):

    # Quadratic default matches lambda_shoot's U = Co' lam / 2:
    if not is_full_cost(cost):
        U = Co.T @ lam / 2.0
        return Co @ U, 0.5 * float(U @ U)

    # General cost: delta = Co dinv(Co' lam), J = g(U):
    U = np.asarray(cost[1](Co.T @ lam)).ravel()
    return Co @ U, float(cost[0](U))

# Derivative of the endpoint map with respect to the costate. This is the
# Gramian itself, which is the whole point of the formulation -- the object
# that prices the cost is also the Jacobian of the map being inverted:
def _ddelta(system, Co, cost, W0, lam):
    return (_gram_at(system, Co, cost, lam) if is_full_cost(cost)
            else 0.5 * W0)

# === RAY BUDGET IDENTITY ===
# Walking out along a costate ray lam = t * lam_hat, stationarity gives
# grad g(u_k) proportional to Co_k' lam_hat and the inverse function theorem
# gives du_k/dt = M_k Co_k' lam_hat, so summing over the horizon,
#
#     dJ/dt = sum_k grad g(u_k)' du_k/dt = t lam_hat' W_g(t) lam_hat
#
# and therefore
#
#     J(t) = integral_0^t tau lam_hat' W_g(tau) lam_hat dtau
#
# The sum over k is the discrete horizon and is exact; the only continuous
# variable is the ray parameter. The Gramian is thus the derivative of the
# budget along the ray, not a global quadratic form. A constant W_g -- which
# is what a quadratic cost gives -- integrates to t^2 lam' W lam / 2, the
# ellipsoid. Anything else does not, and the calibration constant below is
# exactly the error of freezing the integrand:
def _ray_budget(system, Co, cost, W0, lam_hat, t, n_quad=24):

    # Nothing spent at zero radius, since g(0) = 0:
    if t <= 0.0:
        return 0.0

    # Gauss-Legendre on [0, t]. The integrand is smooth in tau away from a
    # deadband crossing and this converges far faster than a uniform rule:
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    tau = 0.5 * t * (xg + 1.0)
    wq = 0.5 * t * wg

    # Accumulate tau * lam_hat' W_g(tau) lam_hat:
    tot = 0.0
    for tk, wk in zip(tau, wq):
        Wk = _ddelta(system, Co, cost, W0, tk * lam_hat)
        tot += wk * tk * float(lam_hat @ Wk @ lam_hat)
    return float(tot)

# === COSTATE SOLVE ===
# Costate placing the endpoint at a requested displacement. The displacement
# is imposed in full, so the off-plane components are zero by construction
# rather than driven to zero as residuals:
def _costate_for(system, Co, cost, W0, target, lam0, iters=40, tol=1e-12):

    lam = np.array(lam0, float)
    scale = max(float(np.linalg.norm(target)), 1e-30)

    # Newton on delta(lam) = target, damped on the residual norm:
    for _ in range(iters):
        delta, J = _probe(system, Co, cost, lam)
        F = delta - target
        nF = float(np.linalg.norm(F))
        if nF <= tol * scale:
            return lam, J

        Jf = _ddelta(system, Co, cost, W0, lam)
        try:
            step = np.linalg.solve(Jf + 1e-13 * np.eye(system.m), F)
        except np.linalg.LinAlgError:
            return lam, J

        # Backtrack, so a poor linearization inside a deadband cannot throw
        # the iterate somewhere it will not come back from:
        t = 1.0
        moved = False
        for _ in range(25):
            lam_t = lam - t * step
            d_t, J_t = _probe(system, Co, cost, lam_t)
            if float(np.linalg.norm(d_t - target)) < nF:
                lam, J, moved = lam_t, J_t, True
                break
            t *= 0.5
        if not moved:
            return lam, J

    delta, J = _probe(system, Co, cost, lam)
    return lam, J

# Operating costate implied by a nominal control. A general cost prices its
# curvature at the iterate, so the Gramian needs a costate; rather than making
# every caller carry one, recover it from the displacement the nominal already
# achieves. The quadratic costate is the starting guess and the general solve
# corrects it. Note the curvature is singular at a zero nominal for any cost
# that is not quadratic, so a general cost needs a nominal that has moved:
def _operating_costate(system, Co, cost, W0, d_op):

    # Quadratic guess inverts d = W0 lam / 2:
    lam0 = _inv_psd(W0) @ (2.0 * d_op)
    if not is_full_cost(cost):
        return lam0

    # Correct it under the general cost:
    lam, _ = _costate_for(system, Co, cost, W0, d_op, lam0)
    return lam

# === BOUNDARY ===
# Radius at which a ray spends the budget. J is increasing in r along a fixed
# direction, so the condition is bracketed and a scalar root find cannot land
# on the antipodal branch -- which a joint solve on a collinearity condition
# can do, since that condition is blind to sign:
def _radius(system, Co, cost, W0, emb, budget, r_guess, lam_warm):

    state = {"lam": np.array(lam_warm, float)}

    # Cost at a radius, warm started from wherever the last call left off.
    # Zero radius is exact rather than solved: no displacement costs nothing,
    # and asking the Newton solve for it only returns whatever the warm start
    # left behind, which breaks the bracket:
    def J_of(r):
        if r <= 0.0:
            return 0.0
        lam, J = _costate_for(system, Co, cost, W0, r * emb, state["lam"])
        state["lam"] = lam
        return J

    # Expand until the budget is bracketed. The ellipsoid overpredicts reach
    # whenever the cost carries a first-order term, so the guess is usually
    # high and it is the upper end that rarely needs growing. The cost at the
    # upper bound is carried out of the loop rather than recomputed:
    r_hi = max(float(r_guess), 1e-12)
    J_hi = J_of(r_hi)
    for _ in range(80):
        if J_hi >= budget:
            break
        r_hi *= 1.6
        J_hi = J_of(r_hi)

    # Bracketed root find on a monotone function. A budget at or below zero
    # has no positive root, and a bracket that fails to straddle means the
    # solve is not returning a monotone cost, so return the bound rather than
    # letting brentq raise:
    if budget <= 0.0 or J_hi - budget <= 0.0:
        lam, _ = _costate_for(system, Co, cost, W0, r_hi * emb, state["lam"])
        return r_hi, lam

    r = float(brentq(lambda rr: J_of(rr) - budget, 0.0, r_hi,
                     xtol=1e-12, rtol=1e-12, maxiter=200))
    lam, _ = _costate_for(system, Co, cost, W0, r * emb, state["lam"])
    return r, lam

# Inverse of a Gramian that may be rank deficient. A plant whose endpoint does
# not respond to a control at first order -- a unicycle steered by turn rate at
# zero control cannot move along its heading -- has a singular W, and that is a
# statement about the plant rather than bad input. Eigenvalues are floored at a
# relative tolerance, so the weak direction inverts to a large but finite value
# and the predicted ellipse collapses to a sliver along the strong direction,
# which is the correct answer: that direction is unreachable to first order:
def _inv_psd(W, rcond=1e-12):

    vals, vecs = np.linalg.eigh(W)
    floor = max(float(vals.max()), 0.0) * rcond

    return vecs @ np.diag(1.0 / np.maximum(vals, max(floor, 1e-300))) @ vecs.T

# Rank of the Gramian at a relative tolerance, reported so a caller can see
# that a degenerate ellipse is degenerate rather than merely thin:
def gramian_rank(W, rcond=1e-12):

    vals = np.linalg.eigvalsh(W)

    return int(np.sum(vals > max(float(vals.max()) * rcond, 1e-300)))

# === QUADRATIC SOLVER ===
# The ellipsoid the Gramian predicts, in closed form from d' P d = 2E. Kept
# separate so the quadratic prediction can be produced and compared without
# tracing anything, and so the tracer can use it as its radial guess:
def ellipse_radii(system, W, budget, dims=(0, 1), n=180, rcond=1e-12):

    # Sampling by polar angle is what makes it comparable to the trace ray by
    # ray. The endpoint is excluded so the closing point is not duplicated,
    # which would leave a zero-length edge in the geometry checks below:
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    Uv = np.vstack([np.cos(th), np.sin(th)])

    # Radius per angle from the sliced inverse Gramian, regularised so a rank
    # deficient plant returns a degenerate ellipse instead of raising:
    P = _inv_psd(W, rcond)[np.ix_(dims, dims)]
    q = np.einsum("in,ij,jn->n", Uv, P, Uv)
    r_ell = np.sqrt(2.0 * budget / np.clip(q, 1e-300, None))

    return th, Uv, r_ell

# === GEOMETRY CHECKS ===
# Both belong to the true set rather than to the tracer: for an even, convex g
# the set is symmetric about the origin and convex, so a tracer that jumps
# branches breaks both while still drawing a plausible looking curve. A
# non-convex g makes the convexity number a property of the cost instead, and
# a negative value is then a result rather than a fault:
def _geometry(pts):

    # Antipodal alignment. Exact only because the angle grid excludes its
    # endpoint and has an even count, so rolling by half the count is a
    # rotation by pi:
    n = len(pts)
    sym = float(np.abs(pts + np.roll(pts, n // 2, axis=0)).max()
                / max(np.abs(pts).max(), 1e-30))

    # Signed turn of consecutive edges, normalised to the largest turn:
    e1 = np.roll(pts, -1, axis=0) - pts
    e2 = np.roll(e1, -1, axis=0)
    crs = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    cvx = float(np.min(crs) / max(np.abs(crs).max(), 1e-30))

    return sym, cvx

# Reachable set on a cost budget, sliced by a coordinate plane:
def reach(system, U, budget, cost=None, lam=None, dims=(0, 1), n=180):

    # Three curves: the true boundary, the ellipsoid the Gramian predicts, and
    # that ellipsoid rescaled by a single calibration constant:
    _, Co = system.endpoint_jac(U)
    W0 = Co @ Co.T

    # Displacement the nominal achieves, which anchors both the operating
    # costate and the calibration below:
    Uf = np.asarray(U).ravel()
    d_op = np.asarray(system.endpoint(U)) - np.asarray(
        system.endpoint(np.zeros_like(np.asarray(U))))

    # A general cost prices curvature at the iterate, so supply the costate
    # from the nominal when the caller did not pass one:
    if lam is None and is_full_cost(cost):
        lam = _operating_costate(system, Co, cost, W0, d_op)

    W = gramian(system, U, cost=cost, lam=lam)
    th, Uv, r_ell = ellipse_radii(system, W, budget, dims=dims, n=n)

    # === CALIBRATION ===
    # A Gramian is a curvature object, so it prices only the second-order part
    # of the cost. A cost with nonzero marginal cost at the origin carries a
    # first-order term the Hessian cannot see, and the ellipsoid then
    # overestimates reach. The achieved endpoint lies on the true boundary at
    # this budget, so requiring the ellipsoid to pass through it fixes the one
    # free scale the family has -- an anchoring condition, not a fit. The cost
    # is called on the flat vector, matching the contract used by _probe:
    J_true = float(cost[0](Uf)) if cost is not None else 0.5 * float(Uf @ Uf)
    Winv = _inv_psd(W)
    J_quad = 0.5 * float(d_op @ (Winv @ d_op))
    c = J_true / max(J_quad, 1e-300)
    r_cal = r_ell / np.sqrt(max(c, 1e-300))

    # === RAY IDENTITY, REPORTED ONLY ===
    # The same constant seen as a quadrature error. The ellipsoid freezes
    # W_g at the operating point and integrates it as a constant, so its
    # budget is the one-point rule for the identity above; the ratio of the
    # exact integral to that rule is why c is identically one for a quadratic
    # cost and departs from one exactly as far as the curvature varies along
    # the ray. Computed as a diagnostic, and not used for the correction:
    lam_op = (np.asarray(lam).ravel() if lam is not None
              else Winv @ d_op)
    t_op = float(np.linalg.norm(lam_op))
    lam_hat = lam_op / max(t_op, 1e-300)
    J_ray = _ray_budget(system, Co, cost, W0, lam_hat, t_op)
    W_op = _ddelta(system, Co, cost, W0, lam_op)
    J_one = 0.5 * t_op ** 2 * float(lam_hat @ W_op @ lam_hat)
    c_quad = J_ray / max(J_one, 1e-300)

    # The identity is exact, so its quadrature reproduces the cost the solve
    # paid whenever the cost, its inverse marginal map, and its curvature all
    # derive from one g in one convention. A nonzero residual is a statement
    # about that consistency, not about the approximation:
    id_err = abs(J_ray - J_true) / max(abs(J_true), 1e-300)

    # === TRACE ===
    # Each ray imposes delta = r * u with r positive, so the direction is
    # fixed outright and only the radius is solved for:
    r_true = np.zeros(n)
    c_dir = np.zeros(n)
    costates = np.zeros((n, system.m))
    off = [i for i in range(system.m) if i not in dims]
    resid = 0.0
    warm = None

    for i in range(n):
        emb = np.zeros(system.m)
        emb[list(dims)] = Uv[:, i]
        if warm is None:
            warm = Winv @ (r_ell[i] * emb)
        r_i, warm = _radius(system, Co, cost, W0, emb, budget, r_ell[i], warm)
        r_true[i] = r_i
        costates[i] = warm

        # Calibration implied by this direction alone. A single constant is
        # exact along the operating ray and extrapolated elsewhere, so the
        # spread of this is the direct test of that extrapolation:
        c_dir[i] = (r_ell[i] / max(r_i, 1e-300)) ** 2

        # Off-plane departure, now a check on the solve rather than on the
        # parameterization, since the plane was imposed:
        delta, _ = _probe(system, Co, cost, warm)
        if off:
            resid = max(resid, float(np.linalg.norm(delta[off])))

    # Relative radial departure, ray by ray:
    def gap(rr):
        rel = np.abs(rr - r_true) / np.maximum(r_true, 1e-30)
        return float(rel.max()), float(rel.mean())

    # Symmetry and convexity of the traced curve:
    pts = (Uv * r_true).T
    sym, cvx = _geometry(pts)

    # Package:
    return dict(theta=th, true=pts, ellipse=(Uv * r_ell).T,
                calibrated=(Uv * r_cal).T, calibration=c,
                calibration_quadrature=c_quad, identity_error=id_err,
                calibration_per_direction=c_dir,
                calibration_spread=(float(c_dir.min()), float(c_dir.max())),
                gap_ellipse=gap(r_ell), gap_calibrated=gap(r_cal),
                costates=costates, budget=float(budget), dims=tuple(dims),
                residual=resid, symmetry=sym, convexity=cvx,
                gramian_rank=gramian_rank(W))

# Vertices of every cell, ordered for drawing or filling:
def hz_vertices(hz):

    out = []
    for cen, G in zip(hz["centers"], hz["generators"]):

        # Put every generator in the upper half plane, then order by angle:
        g = G.T.copy()
        g[g[:, 1] < 0] *= -1.0
        g = g[np.argsort(np.arctan2(g[:, 1], g[:, 0]))]

        # Walk the support in angle order to trace the zonotope hull:
        p = cen - g.sum(axis=0)
        v = [p.copy()]
        for s in (1.0, -1.0):
            for gi in g:
                p = p + s * 2.0 * gi
                v.append(p.copy())
        out.append(np.asarray(v[:-1]))

    return out

# Membership in the union, tested cell by cell:
def hz_contains(hz, pts, tol=1e-9):

    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    inside = np.zeros(len(pts), dtype=bool)

    for cen, G in zip(hz["centers"], hz["generators"]):

        # A two generator zonotope inverts exactly, so membership is a box
        # test in the generator basis. Corners of a cell sit on its own face,
        # so the bound carries a tolerance:
        det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
        if abs(det) < 1e-14:
            continue
        d = pts - cen
        xi0 = (G[1, 1] * d[:, 0] - G[0, 1] * d[:, 1]) / det
        xi1 = (-G[1, 0] * d[:, 0] + G[0, 0] * d[:, 1]) / det
        inside |= (np.abs(xi0) <= 1.0 + tol) & (np.abs(xi1) <= 1.0 + tol)

    return inside

# Area of the union, by rasterising the membership test on a bounding grid:
def hz_area(hz, ng=400):

    V = np.vstack(hz_vertices(hz))
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], ng),
                         np.linspace(lo[1], hi[1], ng), indexing="ij")
    m = hz_contains(hz, np.stack([gx.ravel(), gy.ravel()], axis=1))
    cell = (hi[0] - lo[0]) * (hi[1] - lo[1]) / (ng * ng)

    return float(m.sum() * cell)

# A readable controllability report for benchmarking a system:
def summary(system, U, cost=None, lam=None):

    # Collect the eigenvalues and eigenvectors:
    vals, vecs = eig(system, U, cost=cost, lam=lam)

    # Fetch some common measures:
    ms = dict(min_eig=float(vals[-1]),
              max_eig=float(vals[0]),
              trace=float(np.sum(vals)),
              log_det=float(np.sum(np.log(vals))),
              condition_number=float(vals[0] / vals[-1]))

    # Fetch energy per direction:
    energy, _ = energy_per_direction(system, U, cost=cost, lam=lam)

    # Package report:
    report = dict(
        eigenvalues=vals,
        eigenvectors=vecs,
        energy_per_direction=energy,
        measures=ms,
        weakest_direction=vecs[:, -1],
        strongest_direction=vecs[:, 0])

    # Return the report:
    return report

# Print the controllability report in a readable block:
def print_summary(system, U, name="system", cost=None, lam=None):

    # Gather the report:
    r = summary(system, U, cost=cost, lam=lam)
    ms = r["measures"]

    # Print a titled block of the key metrics:
    print(f"=== reachability report: {name} ===")
    print(f"  endpoint dimension        : {system.m}")
    print(f"  strongest reach eigenvalue: {ms['max_eig']:.4e}")
    print(f"  weakest   reach eigenvalue: {ms['min_eig']:.4e}")
    print(f"  reachable volume (log det): {ms['log_det']:.4f}")
    print(f"  anisotropy (condition no.): {ms['condition_number']:.4e}")

    # Energy per direction is exact only for a quadratic cost:
    tag = "" if cost is None else "  (curvature model)"
    print(f"  energy to move endpoint one unit per principal direction:{tag}")
    for i, e in enumerate(r["energy_per_direction"]):
        print(f"    direction {i}: {e:.4e}")

    # Return the report:
    return r

from scipy import ndimage

# === VALUE FIELD ===
# V(x) is the minimum effort placing the endpoint at x, which lambda_shoot
# returns directly. Its sublevel sets are the reachable sets for EVERY budget
# at once, and because nothing is linearised along the way, holes, bands and
# disconnected pieces survive. This is the only construction here that is not
# restricted to star-shaped or convex sets.
#
# Cost is one shooting solve per query point. Solves are warm started from the
# neighbour just visited, gaps are retried from any converged neighbour, and
# refinement is spent only where a budget contour actually passes, so the
# solve count tracks the length of the contours rather than the area.

# Minimum effort to place the endpoint at a target, through the public solver.
# The target is length system.m, i.e. only the states in system.tidx, so build
# the system with target_idx set to the plotted coordinates and every other
# state is genuinely free rather than pinned at some arbitrary value:
def value_at(engine, system, target, cost=None, warm=None, quiet=True,
             max_it=None, ftol=None):

    import io
    import contextlib

    # ftol is where the runtime is, and it can be loosened without costing
    # accuracy. The inner loop runs until the step falls below ftol, on a
    # fixed point map that barely contracts, and each iteration costs an
    # endpoint Jacobian plus two rollouts. But the value is STATIONARY at the
    # optimum: a control off by eps changes V by order eps squared, so a
    # thousandfold looser step tolerance perturbs V in its sixth digit. The
    # endpoint is not affected at all, since lambda_shoot finishes with a
    # feasibility newton_shoot that puts it back on target regardless.
    #
    # Capping max_it is the unsafe knob by comparison: it truncates solves
    # that were still converging, and each one becomes a hole in the surface.
    # Unreachable targets do not need it -- their trust radius collapses to
    # the floor in a dozen iterations and the unconstrained path breaks out.
    #
    # Most queries near the frontier sit outside the set, where the solver
    # correctly reports it could not satisfy the request. Expected here:
    sink = io.StringIO() if quiet else None
    kw = dict(U0=warm, cost=cost)
    if max_it is not None:
        kw["max_it"] = max_it
    if ftol is not None:
        kw["ftol"] = ftol
    if sink is None:
        U = engine.shooting.lambda_shoot(np.asarray(target, float), **kw)
    else:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            U = engine.shooting.lambda_shoot(np.asarray(target, float), **kw)

    Uf = np.asarray(U, float).ravel()

    # lambda_shoot sets _infeasible when the endpoint was missed or a row is
    # still violated, which is the solver's own verdict and better than any
    # residual test applied from outside:
    if not np.all(np.isfinite(Uf)) or getattr(system, "_infeasible", False):
        return np.inf, None

    J = float(cost[0](Uf)) if cost is not None else 0.5 * float(Uf @ Uf)

    return J, Uf

# Grow the value field outward from the nominal endpoint, and stop at the
# largest budget asked for. Two things make this cheap. The reachable set is
# the image of a connected effort sublevel set, so it is connected and a
# breadth-first expansion reaches all of it without ever solving deep in the
# exterior. And the level sets wanted are strictly inside the reachable set,
# so the expansion halts on cells that merely cost too much rather than on
# cells the solver cannot reach -- a failed solve runs its whole iteration
# budget and returns nothing, so a sweep whose frontier is made of failures
# spends most of its time learning where the set is not.
#
# The frontier is ordered by cost, so cells are solved cheapest first and each
# warm start comes from an adjacent cell that already converged, which is the
# single biggest factor in whether lambda_shoot converges at all.
def value_field(engine, system, U0, ext, ng, budget_max, cost=None,
                dims=(0, 1), quiet=True, max_solves=None, max_it=None,
                ftol=None):

    import heapq

    e0 = np.asarray(system.endpoint(U0)).ravel()
    xs = np.linspace(ext[0], ext[1], ng)
    ys = np.linspace(ext[2], ext[3], ng)
    V = np.full((ng, ng), np.inf)
    ctrl = {}
    seen = np.zeros((ng, ng), bool)
    n = 0
    nfail = 0

    # Seed at the cell nearest the nominal endpoint. searchsorted would round
    # up, and for a plant whose nominal already sits on the boundary -- a unit
    # speed unicycle over a fixed horizon cannot travel further than its own
    # path length, so U0 = 0 lands exactly at the maximum reach -- rounding up
    # puts the seed outside the set and the solver is right to refuse it:
    i0 = int(np.argmin(np.abs(xs - e0[0])))
    j0 = int(np.argmin(np.abs(ys - e0[1])))

    # Order the frontier by the value of the cell it came from, so the cheapest
    # region is filled first and warm starts stay close to their target:
    # The seed is not solved for. U0 already reaches its own endpoint, so its
    # cost is known outright, and asking the solver to reproduce a point it
    # was handed is both wasteful and the one query most likely to sit on the
    # boundary. This mirrors the tracer, where zero radius is exact rather
    # than solved:
    Uf = np.asarray(U0).ravel()
    V[i0, j0] = float(cost[0](Uf)) if cost is not None else 0.5 * float(Uf @ Uf)
    ctrl[(i0, j0)] = Uf
    seen[i0, j0] = True

    # Eight-connected, not four. A set that is thin or lies along a diagonal --
    # which is what an unstable plant produces, since one direction is
    # amplified far more than the other -- can be a single cell wide, and a
    # four-connected frontier cannot cross it:
    STEP = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))

    # Seed a small block rather than one cell. The nominal may sit on the edge
    # of the extent, where a single seed has only two neighbours to offer and
    # the sweep dies before it starts:
    heap = []
    for di, dj in STEP:
        a, b = i0 + di, j0 + dj
        if 0 <= a < ng and 0 <= b < ng and not seen[a, b]:
            seen[a, b] = True
            heapq.heappush(heap, (V[i0, j0], a, b, Uf))

    while heap:
        _, i, j, warm = heapq.heappop(heap)
        v, U = value_at(engine, system, np.array([xs[i], ys[j]]), cost, warm,
                        quiet=quiet, max_it=max_it, ftol=ftol)
        n += 1
        if not np.isfinite(v):
            nfail += 1
        V[i, j] = v
        if not np.isfinite(v):
            continue
        ctrl[(i, j)] = U

        # Past the largest budget the value is recorded but not grown from,
        # which is what keeps the frontier inside the reachable set where the
        # solver converges quickly:
        if v > budget_max:
            continue
        if max_solves is not None and n >= max_solves:
            break

        # Only a converged cell under budget may extend the frontier:
        for di, dj in STEP:
            a, b = i + di, j + dj
            if 0 <= a < ng and 0 <= b < ng and not seen[a, b]:
                seen[a, b] = True
                heapq.heappush(heap, (v, a, b, U))

    # A nominal sitting on the boundary has few reachable neighbours, but if
    # nothing at all converged the request itself is malformed rather than
    # merely tight, so say which of the two it is:
    if len(ctrl) <= 1:
        raise RuntimeError(
            "value_field: %d solves from the nominal and none converged.\n"
            "  nominal endpoint e0 = %s\n"
            "  grid extent          = %s\n"
            "  seed cell            = (%d, %d) of %d, cell size %.3g x %.3g\n"
            "The seed is exact, so this is the request rather than the solver. "
            "A seed on the edge of the extent, or an extent that does not "
            "surround e0, is the usual cause."
            % (n, np.array2string(e0[:2], precision=4),
               np.array2string(np.asarray(ext, float), precision=4),
               i0, j0, ng, xs[1] - xs[0], ys[1] - ys[0]))

    return V, xs, ys, ctrl, n, nfail

# Refine a field where it matters. A cost contour only needs resolution near
# itself: cells deep inside the set and cells well outside it are already
# decided, and solving them again at twice the density buys nothing. Only the
# coarse cells whose neighbourhood straddles one of the budgets are subdivided,
# so the solve count tracks contour length rather than area and the effective
# resolution doubles for a fraction of a uniform sweep:
def refine_field(engine, system, V, xs, ys, ctrl, budgets, cost=None,
                 quiet=True, max_it=None):

    ng = len(xs)
    fine = 2 * ng - 1
    xs2 = np.linspace(xs[0], xs[-1], fine)
    ys2 = np.linspace(ys[0], ys[-1], fine)

    # Carry the coarse values onto the even nodes of the fine grid:
    V2 = np.full((fine, fine), np.nan)
    V2[::2, ::2] = V
    ctrl2 = {(2 * i, 2 * j): U for (i, j), U in ctrl.items()}

    # A coarse cell is near a contour when its 3x3 neighbourhood spans a budget:
    C = np.where(np.isfinite(V), V, np.inf)
    near = np.zeros((ng, ng), bool)
    for b in budgets:
        below = C <= b
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                sh = np.roll(np.roll(below, di, 0), dj, 1)
                near |= below != sh

    # Fine nodes inside a flagged coarse cell, that do not already have a value:
    todo = []
    for i, j in np.argwhere(near):
        for a in (2 * i - 1, 2 * i, 2 * i + 1):
            for b in (2 * j - 1, 2 * j, 2 * j + 1):
                if 0 <= a < fine and 0 <= b < fine and np.isnan(V2[a, b]):
                    todo.append((a, b))
    todo = sorted(set(todo))

    n = 0
    nfail = 0
    for a, b in todo:

        # Warm start from the nearest node already solved:
        warm = None
        for da, db in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1),
                       (1, -1), (-1, 1)):
            if (a + da, b + db) in ctrl2:
                warm = ctrl2[(a + da, b + db)]
                break
        v, U = value_at(engine, system, np.array([xs2[a], ys2[b]]), cost, warm,
                        quiet=quiet, max_it=max_it)
        n += 1
        V2[a, b] = v
        if U is not None:
            ctrl2[(a, b)] = U
        else:
            nfail += 1

    # Everything not solved inherits its nearest known value, which is exact
    # for the interior and the exterior and only approximate near a contour,
    # where the solves just done are:
    miss = np.isnan(V2)
    if miss.any():
        idx = ndimage.distance_transform_edt(miss, return_distances=False,
                                             return_indices=True)
        V2[miss] = V2[tuple(k[miss] for k in idx)]

    return V2, xs2, ys2, ctrl2, n, nfail

# A failed solve is not proof of unreachability, only that this warm start did
# not converge. Retry every gap from a converged neighbour before calling a
# cell unreachable, which is what stops solver noise punching holes in the set:
def repair(engine, system, U0, V, xs, ys, ctrl, cost=None, dims=(0, 1),
           rounds=3, quiet=True, ftol=None):

    e0 = np.asarray(system.endpoint(U0)).ravel()
    ng = len(xs)
    n = 0

    for _ in range(rounds):

        # Only gaps that touch a solved cell are worth retrying; a gap in the
        # deep exterior has no converged neighbour to warm start from and is
        # not part of any level set being drawn:
        gaps = [(i, j) for i, j in np.argwhere(~np.isfinite(V))
                if any((i + di, j + dj) in ctrl
                       for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                      (1, 1), (1, -1), (-1, 1), (-1, -1)))]
        if not gaps:
            break
        fixed = 0
        for i, j in gaps:
            warm = None
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = i + di, j + dj
                if 0 <= a < ng and 0 <= b < ng and (a, b) in ctrl:
                    warm = ctrl[(a, b)]
                    break
            v, U = value_at(engine, system, np.array([xs[i], ys[j]]), cost,
                            warm, quiet=quiet, ftol=ftol)
            n += 1
            if np.isfinite(v):
                V[i, j] = v
                ctrl[(i, j)] = U
                fixed += 1
        if not fixed:
            break

    return V, n

# Sublevel set of the value field at a budget, on a display grid:
def sublevel(V, xs, ys, ext, budget, ng):

    from scipy.interpolate import RegularGridInterpolator

    Vf = np.where(np.isfinite(V), V, 1e18)
    f = RegularGridInterpolator((xs, ys), Vf, bounds_error=False,
                                fill_value=1e18)
    gx = np.linspace(ext[0], ext[1], ng)
    gy = np.linspace(ext[2], ext[3], ng)
    X, Y = np.meshgrid(gx, gy, indexing="ij")

    return f(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(ng, ng) <= budget

# Hybrid zonotope of a sublevel set. Marching squares gives the contours, which
# may be several disconnected loops with holes, and each segment pair spans a
# cell whose enclosing parallelogram is a two generator zonotope:
def hybrid_zonotope(V, xs, ys, budgets):

    from skimage import measure

    levels = []
    centers = []
    gens = []

    for b in sorted(float(x) for x in budgets):
        Vf = np.where(np.isfinite(V), V, 1e18)
        loops = measure.find_contours(Vf, b)
        pts = []
        for lp in loops:
            p = np.stack([np.interp(lp[:, 0], np.arange(len(xs)), xs),
                          np.interp(lp[:, 1], np.arange(len(ys)), ys)], axis=1)
            pts.append(p)
        levels.append(pts)

    # Cells between consecutive budgets on matching contour loops:
    for j in range(len(levels) - 1):
        for a in levels[j]:
            for b in levels[j + 1]:
                m = min(len(a), len(b))
                if m < 3:
                    continue
                ai = a[np.linspace(0, len(a) - 1, m).astype(int)]
                bi = b[np.linspace(0, len(b) - 1, m).astype(int)]
                for i in range(m - 1):
                    quad = np.stack([ai[i], ai[i + 1], bi[i], bi[i + 1]])
                    cen = quad.mean(axis=0)
                    g1 = 0.25 * ((quad[2] + quad[3]) - (quad[0] + quad[1]))
                    g2 = 0.25 * ((quad[1] + quad[3]) - (quad[0] + quad[2]))
                    G = np.stack([g1, g2], axis=1)
                    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
                    if abs(det) < 1e-14:
                        continue

                    # Scale until all four corners are enclosed, the smallest
                    # scaling of this parallelotope that contains the cell:
                    d = quad - cen
                    x0 = (G[1, 1] * d[:, 0] - G[0, 1] * d[:, 1]) / det
                    x1 = (-G[1, 0] * d[:, 0] + G[0, 0] * d[:, 1]) / det
                    s0 = max(float(np.max(np.abs(x0))), 1.0)
                    s1 = max(float(np.max(np.abs(x1))), 1.0)
                    centers.append(cen)
                    gens.append(np.stack([g1 * s0, g2 * s1], axis=1))

    centers = np.asarray(centers)
    gens = np.asarray(gens)

    return dict(centers=centers, generators=gens, levels=levels,
                budgets=sorted(float(x) for x in budgets),
                n_cells=len(centers), n_continuous=2,
                n_binary=int(np.ceil(np.log2(max(len(centers), 2)))))


# Measure what a looser step tolerance costs in value. Solves a sample of
# points twice, once at the loose tolerance and once at the solver default,
# and reports the relative disagreement. V is stationary at the optimum so
# this should be tiny; if it is not, the loosening is not safe on that plant:
def ftol_check(engine, system, U0, ext, cost=None, ftol=1e-3, n=25, seed=0):

    rng = np.random.default_rng(seed)
    e0 = np.asarray(system.endpoint(U0)).ravel()
    rel = []

    for _ in range(n):
        t = np.array([rng.uniform(ext[0], ext[1]), rng.uniform(ext[2], ext[3])])
        v_loose, _ = value_at(engine, system, t, cost, warm=U0, ftol=ftol)
        v_tight, _ = value_at(engine, system, t, cost, warm=U0)
        if np.isfinite(v_loose) and np.isfinite(v_tight) and abs(v_tight) > 0:
            rel.append(abs(v_loose - v_tight) / abs(v_tight))

    if not rel:
        return None

    return dict(n=len(rel), median=float(np.median(rel)),
                worst=float(np.max(rel)))