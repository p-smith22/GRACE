# ============================================================================
# sweep_2d.py -- speedup over IPOPT across state dimension and horizon
# ============================================================================
# Sweeps a chain of decoupled planar quadrotors (nx = 6 per body) against a
# monolithic single-shooting NLP. Each cell is built and solved once to warm the
# compile cache and discard that cost, then solved again for the reported time.
# Cells where either solver fails to reach the reference cost are marked and
# excluded from the map rather than being reported as fast.
# ============================================================================

import json
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import grace

G = 9.81
J_INV = 1.0 / 0.02

# === SWEEP GRID ===
# Total manoeuvre time is held fixed and dt = T_TOTAL / N, so the horizon axis
# varies the discretization rather than the difficulty of the task. With dt
# fixed instead, a short horizon is also a physically harder problem and the
# sweep measures reachability rather than solver scaling.
T_TOTAL = 3.0
N_BODIES = [1, 2, 3, 4, 6]
HORIZONS = [20, 40, 60, 100, 150]
REPEATS = 3

# With the obstacle active, every node carries a keep-out row: GRACE runs its
# augmented-Lagrangian path and transcription carries N inequality constraints,
# so this exercises the constrained machinery on both sides rather than the
# terminal-constraint-only case.
OBSTACLE = True
OBS_XY = (1.6, 1.0)
OBS_R = 0.55

# Positive inside the disc, so g(z, u) <= 0 clears it. One row per body:
def keep_out_terms(z, n_body):
    return [OBS_R ** 2 - ((z[6 * i] - OBS_XY[0]) ** 2
                          + (z[6 * i + 1] - OBS_XY[1]) ** 2)
            for i in range(n_body)]

# === PLANT ===
# n decoupled planar quadrotors, thrust commanded as a deviation from hover:
def make_dynamics(n_body):
    def dynamics(z, u):
        rows = []
        for i in range(n_body):
            zi = z[6 * i:6 * i + 6]
            ui = u[2 * i:2 * i + 2]
            T = G + ui[0]
            rows += [zi[3], zi[4], zi[5],
                     -T * ca.sin(zi[2]),
                     T * ca.cos(zi[2]) - G,
                     J_INV * ui[1]]
        return ca.vertcat(*rows)
    return dynamics

def make_target(n_body):
    tgt = np.zeros(6 * n_body)
    for i in range(n_body):
        tgt[6 * i + 0] = 3.0 + 0.5 * i
        tgt[6 * i + 1] = 2.0
    return tgt

# === IPOPT REFERENCE ===
# Same single-shooting problem, minimum effort to the terminal state:
def ipopt_solve(dynamics, nx, nu, N, z0, target, dt):
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    k1 = dynamics(z, u)
    k2 = dynamics(z + 0.5 * dt * k1, u)
    k3 = dynamics(z + 0.5 * dt * k2, u)
    k4 = dynamics(z + dt * k3, u)
    step = ca.Function("st", [z, u], [z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

    U = ca.MX.sym("U", N * nu)
    Zc = step.mapaccum("roll", N)(ca.DM(z0), ca.reshape(U, nu, N))
    g = Zc[:, -1] - ca.DM(target)
    n_eq = nx

    # Keep-out rows at every node, stacked after the terminal equality:
    if OBSTACLE:
        rows = []
        for k in range(N):
            rows += keep_out_terms(Zc[:, k], nx // 6)
        g = ca.vertcat(g, *rows)

    t0 = time.perf_counter()
    S = ca.nlpsol("ref", "ipopt", dict(x=U, f=ca.dot(U, U), g=g),
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-10),
                       print_time=0))
    t_build = time.perf_counter() - t0

    # One warm call, then the timed calls:
    n_g = g.shape[0]
    lbg = np.concatenate([np.zeros(n_eq), np.full(n_g - n_eq, -np.inf)])
    ubg = np.zeros(n_g)
    args = dict(x0=np.zeros(N * nu), lbg=lbg, ubg=ubg)
    sol = S(**args)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        sol = S(**args)
        ts.append(time.perf_counter() - t0)
    Uv = np.array(sol["x"]).flatten()
    return float(Uv @ Uv), float(np.median(ts)), t_build, S.stats()["success"]

# === IPOPT TRANSCRIPTION BASELINE ===
# States and controls are both decision variables and the dynamics are per-node
# equality constraints. The KKT system is block-banded rather than dense, so
# this is the fair long-horizon baseline: single shooting couples every control
# to the endpoint and its factorization grows with the cube of the horizon.
def ipopt_transcription(dynamics, nx, nu, N, z0, target, dt):
    z = ca.MX.sym("zs", nx)
    u = ca.MX.sym("us", nu)
    k1 = dynamics(z, u)
    k2 = dynamics(z + 0.5 * dt * k1, u)
    k3 = dynamics(z + 0.5 * dt * k2, u)
    k4 = dynamics(z + dt * k3, u)
    step = ca.Function("st", [z, u], [z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

    Zv = [ca.MX.sym(f"z{k}", nx) for k in range(N + 1)]
    Uv = [ca.MX.sym(f"u{k}", nu) for k in range(N)]

    # Initial condition, one defect per interval, then the terminal state:
    gc = [Zv[0] - ca.DM(z0)]
    for k in range(N):
        gc.append(Zv[k + 1] - step(Zv[k], Uv[k]))
    gc.append(Zv[N] - ca.DM(target))
    n_eq = (N + 2) * nx

    # Keep-out rows at every node, stacked after the equalities:
    if OBSTACLE:
        for k in range(N + 1):
            gc += keep_out_terms(Zv[k], nx // 6)

    X = ca.vertcat(*(Zv + Uv))
    Gc = ca.vertcat(*gc)
    f = sum(ca.dot(Uv[k], Uv[k]) for k in range(N))

    t0 = time.perf_counter()
    S = ca.nlpsol("tr", "ipopt", dict(x=X, f=f, g=Gc),
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-10),
                       print_time=0))
    t_build = time.perf_counter() - t0

    ng = Gc.shape[0]
    lbg = np.concatenate([np.zeros(n_eq), np.full(ng - n_eq, -np.inf)])
    args = dict(x0=np.zeros(X.shape[0]), lbg=lbg, ubg=np.zeros(ng))
    sol = S(**args)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        sol = S(**args)
        ts.append(time.perf_counter() - t0)
    xv = np.array(sol["x"]).flatten()
    Ut = xv[(N + 1) * nx:]
    return float(Ut @ Ut), float(np.median(ts)), t_build, S.stats()["success"]

# === GRACE ===
def grace_solve(dynamics, nx, nu, N, z0, target, dt, job):
    cons = ([lambda z, u: ca.vertcat(*keep_out_terms(z, nx // 6))]
            if OBSTACLE else [])
    t0 = time.perf_counter()
    system = grace.build_cached(dynamics, nx=nx, nu=nu, N=N, z0=z0, dt=dt,
                                job=job)
    engine = grace.GRACE(system)
    t_build = time.perf_counter() - t0

    # Warm call first so any residual compile or trace cost is discarded:
    engine.shooting.lambda_shoot(target, constraints=cons)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        U = engine.shooting.lambda_shoot(target, constraints=cons)
        ts.append(time.perf_counter() - t0)
    U = np.asarray(U).flatten()

    # Stationarity from the endpoint Jacobian, as the KKT residual:
    # Worst keep-out violation along the solved trajectory:
    viol = -np.inf
    if OBSTACLE:
        Z = np.asarray(system.rollout(U))
        for i in range(nx // 6):
            d2 = ((Z[:, 6 * i] - OBS_XY[0]) ** 2
                  + (Z[:, 6 * i + 1] - OBS_XY[1]) ** 2)
            viol = max(viol, float((OBS_R ** 2 - d2).max()))

    g, Co = system.endpoint_jac(U)
    lam = np.linalg.solve(Co @ Co.T + 1e-12 * np.eye(nx), Co @ (2.0 * U))
    stat = (np.linalg.norm(2.0 * U - Co.T @ lam)
            / max(np.linalg.norm(2.0 * U), 1e-30))
    end = float(np.linalg.norm(g - system.target(target)))
    return float(U @ U), float(np.median(ts)), t_build, float(stat), end, viol

# === RUN ===
if __name__ == "__main__":
    MODE = "obs" if OBSTACLE else "free"
    results = []
    print(f"total manoeuvre time fixed at {T_TOTAL:.1f} s, dt = T/N")
    print(f"mode: {'obstacle (keep-out at every node)' if OBSTACLE else 'unconstrained'}\n")
    print(f"{'nx':>4}{'N':>6}{'dt':>7}{'vars':>7}{'GRACE':>9}{'shoot':>9}"
          f"{'transcr':>9}{'vs shoot':>10}{'vs trans':>10}"
          f"{'d_cost':>9}{'stat':>9}{'viol':>9}{'ok':>5}")
    for n_body in N_BODIES:
        nx, nu = 6 * n_body, 2 * n_body
        dynamics = make_dynamics(n_body)
        z0 = np.zeros(nx)
        target = make_target(n_body)
        for N in HORIZONS:
            dt = T_TOTAL / N
            try:
                c_ip, t_ip, b_ip, ok_ip = ipopt_solve(dynamics, nx, nu, N,
                                                      z0, target, dt)
                c_tr, t_tr, b_tr, ok_tr = ipopt_transcription(
                    dynamics, nx, nu, N, z0, target, dt)
                c_gr, t_gr, b_gr, stat, end, viol = grace_solve(
                    dynamics, nx, nu, N, z0, target, dt,
                    job=f"sweep_{MODE}_{n_body}_{N}")
                dc = abs(c_gr - c_ip) / max(abs(c_ip), 1e-30)
                dct = abs(c_tr - c_ip) / max(abs(c_ip), 1e-30)

                # A cell only counts if all three agree and GRACE is stationary:
                ok = bool(ok_ip and ok_tr and dc < 1e-4 and dct < 1e-4
                          and stat < 1e-4 and end < 1e-6
                          and (not OBSTACLE or viol < 1e-6))
                results.append(dict(n_body=n_body, nx=nx, nu=nu, N=N, dt=dt,
                                    vars=N * nu, t_grace=t_gr, t_ipopt=t_ip,
                                    t_transcription=t_tr,
                                    build_grace=b_gr, build_ipopt=b_ip,
                                    build_transcription=b_tr,
                                    cost_grace=c_gr, cost_ipopt=c_ip,
                                    cost_transcription=c_tr,
                                    d_cost=dc, d_cost_tr=dct,
                                    stat=stat, end=end, viol=viol,
                                    obstacle=OBSTACLE, ok=ok))
                print(f"{nx:>4}{N:>6}{dt:>7.3f}{N * nu:>7}{t_gr * 1e3:>9.2f}"
                      f"{t_ip * 1e3:>9.2f}{t_tr * 1e3:>9.2f}"
                      f"{t_ip / t_gr:>10.2f}{t_tr / t_gr:>10.2f}"
                      f"{dc:>9.1e}{stat:>9.1e}"
                      f"{(viol if OBSTACLE else 0.0):>9.1e}{str(ok):>5}")
            except Exception as e:
                print(f"{nx:>4}{N:>6}{dt:>7.3f}  FAILED: {type(e).__name__}: {e}")
                results.append(dict(n_body=n_body, nx=nx, nu=nu, N=N,
                                    ok=False, error=str(e)))

    with open(f"sweep_2d_{MODE}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote sweep_2d_{MODE}.json")

    # === PLOT ===
    res = results
    nxs = sorted({r["nx"] for r in res})
    Ns = sorted({r["N"] for r in res})
    bad = []
    Zs = np.full((len(nxs), len(Ns)), np.nan)
    Zt = np.full((len(nxs), len(Ns)), np.nan)

    for r in res:
        i, j = nxs.index(r["nx"]), Ns.index(r["N"])
        if r.get("ok"):
            Zs[i, j] = r["t_ipopt"] / r["t_grace"]
            Zt[i, j] = r["t_transcription"] / r["t_grace"]
        else:
            bad.append((j, i))

    # Shared colour scale so the two baselines are directly comparable:
    fin = np.concatenate([Zs[np.isfinite(Zs)], Zt[np.isfinite(Zt)]])
    vmax = max(fin.max(), 1.0 / fin.min()) if fin.size else 10.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, Z, lab in ((axes[0], Zs, "single shooting"),
                       (axes[1], Zt, "transcription")):
        im = ax.imshow(Z, origin="lower", cmap="RdYlGn", aspect="auto",
                       norm=LogNorm(vmin=1.0 / vmax, vmax=vmax))
        for i in range(len(nxs)):
            for j in range(len(Ns)):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.1f}x", ha="center", va="center",
                            fontsize=10, color="black")
        for j, i in bad:
            ax.text(j, i, "x", ha="center", va="center", fontsize=14,
                    color="0.4")
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(Ns)
        ax.set_yticks(range(len(nxs)))
        ax.set_yticklabels(nxs)
        ax.set_xlabel("horizon N")
        ax.set_ylabel("state dimension nx")
        ax.set_title(f"vs IPOPT {lab}")
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"GRACE solve-time speedup  "
                 f"({T_TOTAL:.0f} s manoeuvre, dt = T/N, "
                 f"{'obstacle' if OBSTACLE else 'unconstrained'}; "
                 f"build excluded; x = solvers disagreed)")
    fig.tight_layout()
    fig.savefig(f"figures/sweep_2d_{MODE}.png", dpi=140, bbox_inches="tight")
    print(f"saved figures/sweep_2d_{MODE}.png")

    ok = [r for r in res if r.get("ok")]
    if ok:
        bs = max(ok, key=lambda r: r["t_ipopt"] / r["t_grace"])
        bt = max(ok, key=lambda r: r["t_transcription"] / r["t_grace"])
        print(f"cells solved: {len(ok)}/{len(res)}")
        print(f"best vs single shooting: nx={bs['nx']} N={bs['N']} -> "
              f"{bs['t_ipopt'] / bs['t_grace']:.2f}x")
        print(f"best vs transcription  : nx={bt['nx']} N={bt['N']} -> "
              f"{bt['t_transcription'] / bt['t_grace']:.2f}x")