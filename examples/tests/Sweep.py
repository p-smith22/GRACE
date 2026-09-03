# ============================================================================
# sweep.py -- where the costate parameterization pays: horizon and dimension
# ============================================================================
# GRACE solves p equations in p unknowns whatever N is, so its work scales
# with the constrained endpoint dimension rather than with the horizon. A
# direct NLP carries N*nu decision variables and pays for every one. This
# sweeps both axes on the same plant against two references: IPOPT single
# shooting, which solves the same problem GRACE does, and IPOPT direct
# transcription, which is the strong baseline whose sparse KKT system scales
# far better than single shooting and is the honest comparison at large nx.
#
# Plant: a chain of masses joined by hardening springs, actuated only at the
# two ends. nx = 2*n_mass scales the state, N scales the horizon, nu stays at
# two throughout, so the underactuation gets worse as the chain grows and the
# problem does not become trivially decoupled.
# ============================================================================
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace

# === PLANT ===
K_LIN, K_CUB, C_DMP, M_MASS = 4.0, 1.5, 0.25, 1.0

# Chain of n masses, springs between neighbours, forces on the first and last:
def make_chain(n):

    def dynamics(z, u):
        q = z[:n]
        v = z[n:]
        acc = []

        # Net spring and damper force on each mass from its neighbours:
        for i in range(n):
            f = 0.0
            if i > 0:
                d = q[i] - q[i - 1]
                f = f - (K_LIN * d + K_CUB * d ** 3) - C_DMP * (v[i] - v[i - 1])
            if i < n - 1:
                d = q[i + 1] - q[i]
                f = f + (K_LIN * d + K_CUB * d ** 3) + C_DMP * (v[i + 1] - v[i])

            # Only the end masses are actuated:
            if i == 0:
                f = f + u[0]
            if i == n - 1:
                f = f + u[1]
            acc.append(f / M_MASS)

        return ca.vertcat(v, ca.vertcat(*acc))

    return dynamics

# Rest-to-rest transfer: every mass displaced, every velocity back to zero:
def problem(n):
    z0 = np.zeros(2 * n)
    tgt = np.concatenate([np.linspace(0.6, 0.3, n), np.zeros(n)])
    return z0, tgt

# === REFERENCES ===
# One RK4 step, shared by both IPOPT formulations:
def _step(dyn, nx, nu, dt):
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    k1 = dyn(z, u)
    k2 = dyn(z + 0.5 * dt * k1, u)
    k3 = dyn(z + 0.5 * dt * k2, u)
    k4 = dyn(z + dt * k3, u)
    return ca.Function("st", [z, u], [z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)])

# Single shooting: controls only, states eliminated by the rollout. Same
# problem GRACE solves, so the comparison is like for like:
def ipopt_shoot(dyn, nx, nu, N, dt, z0, tgt):
    step = _step(dyn, nx, nu, dt)
    U = ca.MX.sym("U", N * nu)
    Zc = step.mapaccum("roll", N)(ca.DM(z0), ca.reshape(U, nu, N))
    g = Zc[:, -1] - ca.DM(tgt)
    S = ca.nlpsol("ss", "ipopt", dict(x=U, f=ca.dot(U, U), g=g),
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-8,
                                  max_iter=400), print_time=0))
    t0 = time.perf_counter()
    sol = S(x0=np.zeros(N * nu), lbg=np.zeros(nx), ubg=np.zeros(nx))
    t = time.perf_counter() - t0
    return (float(ca.dot(sol["x"], sol["x"])), t, S.stats()["success"],
            S.stats()["iter_count"])

# Direct transcription: states and controls both decision variables, dynamics
# as per node equalities. The KKT system is sparse and banded, which is why
# this is the baseline that matters at large nx:
def ipopt_trans(dyn, nx, nu, N, dt, z0, tgt):
    step = _step(dyn, nx, nu, dt)
    Zv = [ca.MX.sym(f"z{k}", nx) for k in range(N + 1)]
    Uv = [ca.MX.sym(f"u{k}", nu) for k in range(N)]
    gc = [Zv[0] - ca.DM(z0)]
    for k in range(N):
        gc.append(Zv[k + 1] - step(Zv[k], Uv[k]))
    gc.append(Zv[N] - ca.DM(tgt))
    X = ca.vertcat(*(Zv + Uv))
    G = ca.vertcat(*gc)
    f = sum(ca.dot(Uv[k], Uv[k]) for k in range(N))
    S = ca.nlpsol("tr", "ipopt", dict(x=X, f=f, g=G),
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-8,
                                  max_iter=400), print_time=0))
    ng = G.shape[0]
    t0 = time.perf_counter()
    sol = S(x0=np.zeros(X.shape[0]), lbg=np.zeros(ng), ubg=np.zeros(ng))
    t = time.perf_counter() - t0
    xv = np.array(sol["x"]).flatten()
    Ut = xv[(N + 1) * nx:]
    return float(Ut @ Ut), t, S.stats()["success"], S.stats()["iter_count"]

# GRACE, timed on the solve alone. The build is a one-off compile and is
# reported separately rather than folded into the comparison:
def grace_solve(dyn, nx, nu, N, dt, z0, tgt, job):
    t0 = time.perf_counter()
    system = grace.build_cached(dyn, nx=nx, nu=nu, N=N, z0=z0, dt=dt, job=job)
    t_build = time.perf_counter() - t0
    engine = grace.GRACE(system)
    t0 = time.perf_counter()
    U = np.asarray(engine.shooting.lambda_shoot(tgt)).flatten()
    t = time.perf_counter() - t0
    err = float(np.linalg.norm(np.asarray(system.endpoint(U))
                               - np.asarray(system.target(tgt))))
    return float(U @ U), t, err, t_build

# One point of either sweep:
def run(n_mass, N, T=6.0, tag=""):
    dyn = make_chain(n_mass)
    nx, nu = 2 * n_mass, 2
    z0, tgt = problem(n_mass)
    dt = T / N
    job = f"chain_n{n_mass}_N{N}"

    cg, tg, eg, tb = grace_solve(dyn, nx, nu, N, dt, z0, tgt, job)
    cs, ts, oks, its = ipopt_shoot(dyn, nx, nu, N, dt, z0, tgt)
    ct, tt, okt, itt = ipopt_trans(dyn, nx, nu, N, dt, z0, tgt)

    # Cost gap against the best converged reference, so a fast wrong answer
    # cannot look like a win:
    refs = [c for c, ok in ((cs, oks), (ct, okt)) if ok]
    gap = (cg / min(refs) - 1.0) * 100.0 if refs else np.nan

    print(f"{tag:<10}{nx:>5}{N:>6}{tg:>9.3f}{ts:>9.3f}{tt:>9.3f}"
          f"{ts / max(tg, 1e-9):>8.1f}x{tt / max(tg, 1e-9):>8.1f}x"
          f"{gap:>9.2f}%{eg:>10.1e}{tb:>8.2f}")
    return dict(nx=nx, N=N, t_g=tg, t_s=ts, t_t=tt, gap=gap, err=eg,
                ok_s=oks, ok_t=okt, its=its, itt=itt, t_build=tb)

# === RUN ===
if __name__ == "__main__":
    hdr = (f"{'':<10}{'nx':>5}{'N':>6}{'GRACE':>9}{'IP ss':>9}{'IP tr':>9}"
           f"{'vs ss':>9}{'vs tr':>9}{'cost gap':>10}{'endpt':>10}{'build':>8}")

    # Horizon sweep at fixed dimension. Maneuver time is held fixed and only
    # the discretization changes, so the physical problem is identical at
    # every point and only the decision vector grows:
    print("=== horizon sweep, 6 masses (nx = 12), fixed 6 s maneuver ===")
    print(hdr)
    Ns = [20, 40, 80, 160, 320]
    hor = [run(6, n, tag="horizon") for n in Ns]

    # Dimension sweep at fixed horizon. nu stays at two, so the chain gets
    # progressively more underactuated as it grows:
    print(f"\n=== dimension sweep, N = 60, chain of 2 to 24 masses ===")
    print(hdr)
    ns = [2, 4, 8, 12]
    dim = [run(n, 60, tag="dim") for n in ns]

    # === PLOT ===
    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.6))

    ax[0].loglog(Ns, [d["t_g"] for d in hor], "o-", color="crimson", lw=2.0,
                 label="GRACE")
    ax[0].loglog(Ns, [d["t_s"] for d in hor], "o-", color="steelblue", lw=2.0,
                 label="IPOPT single shooting")
    ax[0].loglog(Ns, [d["t_t"] for d in hor], "o--", color="seagreen", lw=2.0,
                 label="IPOPT transcription")
    ax[0].set_xlabel("horizon steps $N$ (maneuver time fixed)")
    ax[0].set_ylabel("solve time [s]")
    ax[0].set_title("Horizon, $n_x = 12$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    nxs = [d["nx"] for d in dim]
    ax[1].loglog(nxs, [d["t_g"] for d in dim], "o-", color="crimson", lw=2.0,
                 label="GRACE")
    ax[1].loglog(nxs, [d["t_s"] for d in dim], "o-", color="steelblue", lw=2.0,
                 label="IPOPT single shooting")
    ax[1].loglog(nxs, [d["t_t"] for d in dim], "o--", color="seagreen", lw=2.0,
                 label="IPOPT transcription")
    ax[1].set_xlabel("state dimension $n_x$ ($n_u = 2$, $N = 60$)")
    ax[1].set_ylabel("solve time [s]")
    ax[1].set_title("Dimension, fixed horizon")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    # Speedup against both references on one axis, so the crossovers are
    # visible rather than inferred:
    ax[2].semilogx(Ns, [d["t_s"] / max(d["t_g"], 1e-9) for d in hor], "o-",
                   color="steelblue", lw=2.0, label="horizon, vs shooting")
    ax[2].semilogx(Ns, [d["t_t"] / max(d["t_g"], 1e-9) for d in hor], "o--",
                   color="seagreen", lw=2.0, label="horizon, vs transcription")
    ax[2].semilogx(nxs, [d["t_s"] / max(d["t_g"], 1e-9) for d in dim], "s-",
                   color="steelblue", lw=1.4, alpha=0.55,
                   label="dimension, vs shooting")
    ax[2].semilogx(nxs, [d["t_t"] / max(d["t_g"], 1e-9) for d in dim], "s--",
                   color="seagreen", lw=1.4, alpha=0.55,
                   label="dimension, vs transcription")
    ax[2].axhline(1.0, color="k", lw=1.0)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("$N$ or $n_x$")
    ax[2].set_ylabel("speedup, above 1 favours GRACE")
    ax[2].set_title("Where the parameterization pays")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("figures/tests/sweep.png", dpi=140, bbox_inches="tight")
    print("\nsaved figures/tests/sweep.png")