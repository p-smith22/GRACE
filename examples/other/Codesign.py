# Profile codesign and benchmark it against a monolithic IPOPT co-design NLP,
# with one-time build cost reported separately from per-front-point solve cost.

import time
import numpy as np
import casadi as ca
from scipy.optimize import minimize
import grace.codesign.codesign as cd

# === LAMBDA_SHOOT COUNTER ===
_real = cd.lambda_shoot
stats = dict(n=0, t=0.0, times=[])

def counted(system, zt, U0=None, **kw):
    t0 = time.perf_counter()
    out = _real(system, zt, U0=U0, **kw)
    dt = time.perf_counter() - t0
    stats["n"] += 1
    stats["t"] += dt
    stats["times"].append(dt)
    return out

cd.lambda_shoot = counted

# === FAMILY BUILD TIMER ===
# codesign builds its own family internally, so time that call rather than
# subtracting a separate standalone build:
_real_build = cd._build_param_family
build_stats = dict(n=0, t=0.0)

def timed_build(*a, **kw):
    t0 = time.perf_counter()
    out = _real_build(*a, **kw)
    build_stats["n"] += 1
    build_stats["t"] += time.perf_counter() - t0
    return out

cd._build_param_family = timed_build

# === PLANT ===
def dynamics(z, u, p):
    return ca.vertcat(z[2], z[3], p * u[0], p * u[1] - 9.81)

def objective(p):
    pv = float(p)
    return 120.0 * pv ** 3 + 900.0 / (1.0 + np.exp(-(pv - 1.20) / 0.03))

def objective_ca(p):
    return 120.0 * p ** 3 + 900.0 / (1.0 + ca.exp(-(p - 1.20) / 0.03))

# === PROBLEM SETUP ===
nx, nu, N, dt = 4, 2, 40, 0.05
z0 = np.zeros(nx)
target = np.array([5.0, 5.0, 0.0, 0.0])
p0 = 1.5
p_bounds = (0.5, 2.0)
weights = np.linspace(0.0, 1.0, 21)

# === MONOLITHIC BASELINE ===
def build_monolithic(norms):
    U = ca.MX.sym("U", N * nu)
    p = ca.MX.sym("p", 1)
    t = ca.MX.sym("t", 1)
    w1 = ca.MX.sym("w1", 1)
    w2 = ca.MX.sym("w2", 1)

    z = ca.DM(z0)
    for k in range(N):
        u = U[k * nu:(k + 1) * nu]
        k1 = dynamics(z, u, p)
        k2 = dynamics(z + 0.5 * dt * k1, u, p)
        k3 = dynamics(z + 0.5 * dt * k2, u, p)
        k4 = dynamics(z + dt * k3, u, p)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    C = ca.dot(U, U)
    D = objective_ca(p)
    Chat = (C - norms["C_id"]) / norms["C_rng"]
    Dhat = (D - norms["D_id"]) / norms["D_rng"]

    # Pure Chebyshev in epigraph form, plus a small control-effort term so the
    # inner problem is min-norm like GRACE's -- without it the objective is flat
    # in U whenever the design term attains the max, and IPOPT stops anywhere:
    g = ca.vertcat(z - ca.DM(target), t - w1 * Chat, t - w2 * Dhat)
    nlp = dict(x=ca.vertcat(U, p, t), p=ca.vertcat(w1, w2),
               f=t + 1e-3 * Chat, g=g)
    opts = dict(ipopt=dict(print_level=0, sb="yes", tol=1e-8), print_time=0)
    solver = ca.nlpsol("mono", "ipopt", nlp, opts)

    lbx = np.r_[np.full(N * nu, -np.inf), p_bounds[0], 0.0]
    ubx = np.r_[np.full(N * nu, np.inf), p_bounds[1], np.inf]
    lbg = np.r_[np.zeros(nx), np.zeros(2)]
    ubg = np.r_[np.zeros(nx), np.full(2, np.inf)]
    return solver, lbx, ubx, lbg, ubg

# === REPORT ===
def report(n_weights, solve_time):
    t = np.array(stats["times"]) * 1e3
    print(f"\n[prof] lambda_shoot calls : {stats['n']}")
    print(f"[prof] calls per weight   : {stats['n'] / max(n_weights, 1):.1f}")
    print(f"[prof] time in shoot      : {stats['t']:.3f} s "
          f"({stats['t'] / max(solve_time, 1e-12) * 100:.0f}% of solve time)")
    print(f"[prof] per call ms        : mean {t.mean():.2f}  "
          f"median {np.median(t):.2f}  min {t.min():.2f}  max {t.max():.2f}")

if __name__ == "__main__":

    # --- framework: full call; internal build timed by the wrapper ---
    build_stats["n"], build_stats["t"] = 0, 0.0
    t0 = time.perf_counter()
    _, _, pareto, sweep = cd.codesign(
        dynamics, nx, nu, N, z0, dt, target, "thrust_gain", objective,
        p0, p_bounds, weights=weights, norm="cheby", n_anchor=9,
        plot=False)
    t_frame_total = time.perf_counter() - t0
    t_build_frame = build_stats["t"]
    t_frame_solve = t_frame_total - t_build_frame

    # --- reproduce codesign's ideal point for the baseline ---
    Cg = np.array([s["cost"] for s in sweep])
    Dg = np.array([s["objective"] for s in sweep])
    C_id = Cg.min() - 0.01 * np.ptp(Cg)
    D_id = Dg.min() - 0.01 * np.ptp(Dg)
    norms = dict(C_id=C_id, D_id=D_id,
                 C_rng=max(Cg[int(np.argmin(Dg))] - C_id, 1e-12),
                 D_rng=max(Dg[int(np.argmin(Cg))] - D_id, 1e-12))

    # --- monolithic: build and solve timed separately ---
    t0 = time.perf_counter()
    solver, lbx, ubx, lbg, ubg = build_monolithic(norms)
    t_build_mono = time.perf_counter() - t0

    x_guess = np.r_[np.zeros(N * nu), p0, 1.0]
    mono, t_mono_solve = [], 0.0
    for w in weights:
        t1 = time.perf_counter()
        sol = solver(x0=x_guess, p=[1.0 - w, w],
                     lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        t_mono_solve += time.perf_counter() - t1
        x_guess = np.array(sol["x"]).flatten()
        Uv, pv = x_guess[:N * nu], float(x_guess[N * nu])
        mono.append(dict(weight=float(w), param=pv, cost=float(Uv @ Uv),
                         objective=objective(pv)))

    # --- scipy SLSQP baseline, AD derivatives from CasADi ---
    t0 = time.perf_counter()
    Us = ca.MX.sym("Us", N * nu)
    ps = ca.MX.sym("ps", 1)
    a1 = ca.MX.sym("a1", 1)
    a2 = ca.MX.sym("a2", 1)
    zs = ca.DM(z0)
    for k in range(N):
        uk = Us[k * nu:(k + 1) * nu]
        q1 = dynamics(zs, uk, ps)
        q2 = dynamics(zs + 0.5 * dt * q1, uk, ps)
        q3 = dynamics(zs + 0.5 * dt * q2, uk, ps)
        q4 = dynamics(zs + dt * q3, uk, ps)
        zs = zs + (dt / 6.0) * (q1 + 2 * q2 + 2 * q3 + q4)
    Cs = (ca.dot(Us, Us) - norms["C_id"]) / norms["C_rng"]
    Ds = (objective_ca(ps) - norms["D_id"]) / norms["D_rng"]
    fs = ca.fmax(a1 * Cs, a2 * Ds) + 1e-3 * Cs
    y = ca.vertcat(Us, ps)
    F_f = ca.Function("f", [y, a1, a2], [fs])
    F_jf = ca.Function("jf", [y, a1, a2], [ca.jacobian(fs, y)])
    F_c = ca.Function("c", [y], [zs - ca.DM(target)])
    F_jc = ca.Function("jc", [y], [ca.jacobian(zs - ca.DM(target), y)])
    t_build_slsqp = time.perf_counter() - t0

    y0 = np.r_[np.zeros(N * nu), p0]
    slsqp, t_slsqp_solve = [], 0.0
    for w in weights:
        aa, bb = 1.0 - w, w
        t1 = time.perf_counter()
        r = minimize(lambda v: float(F_f(v, aa, bb)), y0, method="SLSQP",
                     jac=lambda v: np.array(F_jf(v, aa, bb)).flatten(),
                     constraints=[dict(
                         type="eq",
                         fun=lambda v: np.array(F_c(v)).flatten(),
                         jac=lambda v: np.array(F_jc(v)))],
                     bounds=[(None, None)] * (N * nu) + [p_bounds],
                     options=dict(maxiter=300, ftol=1e-9))
        t_slsqp_solve += time.perf_counter() - t1
        y0 = r.x
        slsqp.append(dict(param=float(y0[N * nu]),
                          cost=float(y0[:N * nu] @ y0[:N * nu])))

    # === RESULTS ===
    nw = len(weights)
    pf = np.array([f["param"] for f in pareto])
    cf = np.array([f["cost"] for f in pareto])
    pm = np.array([f["param"] for f in mono])
    cm = np.array([f["cost"] for f in mono])
    print("\n=== BUILD (one-time) ===")
    print(f"  framework family (jit)   : {t_build_frame:6.3f} s "
          f"({build_stats['n']} build(s))")
    print(f"  monolithic nlpsol        : {t_build_mono:6.3f} s")
    print(f"  slsqp AD functions       : {t_build_slsqp:6.3f} s")

    print("\n=== SOLVE (per front point) ===")
    print(f"  {'framework (GRACE)':<22}{t_frame_solve:7.3f} s "
          f"{t_frame_solve / nw * 1e3:8.2f} ms/pt   1.00x")
    print(f"  {'monolithic IPOPT':<22}{t_mono_solve:7.3f} s "
          f"{t_mono_solve / nw * 1e3:8.2f} ms/pt  "
          f"{t_mono_solve / max(t_frame_solve, 1e-12):5.2f}x")
    print(f"  {'scipy SLSQP':<22}{t_slsqp_solve:7.3f} s "
          f"{t_slsqp_solve / nw * 1e3:8.2f} ms/pt  "
          f"{t_slsqp_solve / max(t_frame_solve, 1e-12):5.2f}x")

    print("\n=== TOTAL (build + solve) ===")
    print(f"  framework  : {t_build_frame + t_frame_solve:6.3f} s")
    print(f"  monolithic : {t_build_mono + t_mono_solve:6.3f} s")
    print(f"  slsqp      : {t_build_slsqp + t_slsqp_solve:6.3f} s")

    ps_ = np.array([f["param"] for f in slsqp])
    print(f"\ndesign agreement IPOPT vs SLSQP: max |dp| = "
          f"{np.abs(np.sort(pm) - np.sort(ps_)).max():.4f}")

    # Both methods solve the same scalarized problem at each weight, so pair
    # them directly -- interpolating across a convex front biases the chord up:
    # Both methods return a design; score both on the same min-effort curve
    # C*(p) so the comparison is not confounded by the NLP's free control:
    fam_ref = cd._build_param_family(dynamics, nx, nu, N, z0, dt, "thrust_gain", 1)
    zt_ref = np.asarray(target, float)[fam_ref["tidx"]]

    def cstar(pv, _cache={}):
        k = round(float(pv), 9)
        if k not in _cache:
            sp = cd._PinnedSystem(fam_ref, nx, nu, N, z0, dt, float(pv))
            Uv = _real(sp, zt_ref, U0=None)
            _cache[k] = float(Uv @ Uv)
        return _cache[k]

    mw = {round(f["weight"], 12): f for f in mono}
    pairs = [(f, mw[round(f["weight"], 12)]) for f in pareto
             if round(f["weight"], 12) in mw]
    pf = np.array([a["param"] for a, _ in pairs])
    pm_paired = np.array([b["param"] for _, b in pairs])
    cf = np.array([cstar(v) for v in pf])
    interp = np.array([cstar(v) for v in pm_paired])
    rel = np.abs(cf - interp) / np.maximum(interp, 1e-12)
    print(f"\ndesign agreement GRACE vs IPOPT: max |dp| = "
          f"{np.abs(pf - pm_paired).max():.5f}")
    print(f"\ncost agreement: max {rel.max() * 100:.3f}%, "
          f"mean {rel.mean() * 100:.3f}%")

    # Locate the worst-agreeing point -- usually a design sitting on a bound:
    iw = int(np.argmax(rel))
    at_bound = (abs(pf[iw] - p_bounds[0]) < 1e-6
                or abs(pf[iw] - p_bounds[1]) < 1e-6)
    print(f"  worst point: p={pf[iw]:.4f} "
          f"({'ON BOUND' if at_bound else 'interior'}), "
          f"GRACE {cf[iw]:.1f} @ p={pf[iw]:.4f} vs "
          f"IPOPT {interp[iw]:.1f} @ p={pm_paired[iw]:.4f}")
    print("  points with >0.5% disagreement:")
    for i in np.where(rel > 0.005)[0]:
        print(f"    p={pf[i]:.4f}  rel={rel[i] * 100:6.3f}%  "
              f"GRACE {cf[i]:9.1f}  IPOPT {interp[i]:9.1f}")

    # === PLOT ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16, 5))

    ax.plot([s_["objective"] for s_ in sweep], [s_["cost"] for s_ in sweep],
            "-", color="0.75", lw=1.4, label="reference sweep")
    ax.plot([f["objective"] for f in mono], [f["cost"] for f in mono],
            "o", color="0.35", ms=9, mfc="none", mew=1.5, label="IPOPT")
    ax.plot([f["objective"] for f in pareto], [f["cost"] for f in pareto],
            "^", color="steelblue", ms=7, label="GRACE")
    ax.set_xlabel("design objective")
    ax.set_ylabel("control effort")
    ax.set_title("Front agreement")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    bx.semilogy(pf, np.maximum(rel * 100, 1e-6), "o-", color="steelblue", ms=5)
    bx.axhline(0.5, color="crimson", ls="--", lw=1, label="0.5%")
    bx.set_xlabel("design parameter p")
    bx.set_ylabel("cost disagreement vs IPOPT (%)")
    bx.set_title("Per-point agreement")
    bx.legend(fontsize=9)
    bx.grid(alpha=0.3, which="both")

    names = ["GRACE", "IPOPT", "SLSQP"]
    solve = [t_frame_solve / nw * 1e3, t_mono_solve / nw * 1e3,
             t_slsqp_solve / nw * 1e3]
    build = [t_build_frame / nw * 1e3, t_build_mono / nw * 1e3,
             t_build_slsqp / nw * 1e3]
    cols = ["steelblue", "0.45", "darkorange"]
    cx.bar(names, solve, color=cols, width=0.6, label="solve")
    cx.bar(names, build, bottom=solve, color=cols, width=0.6, alpha=0.4,
           hatch="//", label="build (amortized)")
    for i, (s_, b_) in enumerate(zip(solve, build)):
        cx.text(i, s_ + b_, f"{s_:.2f}", ha="center", va="bottom", fontsize=9)
    cx.set_ylabel("ms per front point")
    cx.set_title("Cost per front point")
    cx.legend(fontsize=9)
    cx.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("figures/benchmark.png", dpi=140, bbox_inches="tight")
    print("\nsaved figures/benchmark.png")

    report(nw, t_frame_solve)

    # --- framework again with jit disabled, to price the compile ---
    cd._build_param_family = lambda *a, **kw: timed_build(
        *a, **{**kw, "jit": False})
    build_stats["n"], build_stats["t"] = 0, 0.0
    stats["n"], stats["t"], stats["times"] = 0, 0.0, []
    t0 = time.perf_counter()
    cd.codesign(dynamics, nx, nu, N, z0, dt, target, "thrust_gain", objective,
                p0, p_bounds, weights=weights, norm="cheby", n_anchor=9,
                plot=False)
    t_nojit_total = time.perf_counter() - t0
    t_nojit_build = build_stats["t"]
    t_nojit_solve = t_nojit_total - t_nojit_build

    print("\n=== JIT ON vs OFF (framework) ===")
    print(f"  jit=True   build {t_build_frame:6.3f} s  solve "
          f"{t_frame_solve / nw * 1e3:6.2f} ms/pt  total "
          f"{t_build_frame + t_frame_solve:6.3f} s")
    print(f"  jit=False  build {t_nojit_build:6.3f} s  solve "
          f"{t_nojit_solve / nw * 1e3:6.2f} ms/pt  total "
          f"{t_nojit_build + t_nojit_solve:6.3f} s")