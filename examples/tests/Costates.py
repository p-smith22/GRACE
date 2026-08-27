# ============================================================================
# costate_comparison.py -- states and costates, GRACE vs IPOPT vs PMP
# ============================================================================
# GRACE's Gramian solve produces a terminal costate directly:
#     lam = (Co R^-1 Co^T)^-1 (Co R^-1 grad - delta)
# This checks that costate against three independent references:
#   1. IPOPT's Lagrange multiplier on the same terminal constraint (mu = -lam)
#   2. the discrete adjoint recursion  lam_k = A_k^T lam_{k+1}
#   3. Hamiltonian stationarity        R u_k = B_k^T lam_{k+1}
# Agreement means the two methods find the same primal *and* dual solution,
# which a runtime comparison cannot show.
# ============================================================================

import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace

# === PROBLEM ===
# Planar quadrotor with thrust commanded as a deviation from hover:
G = 9.81
J_INV = 1.0 / 0.02
NX, NU, N, DT = 6, 2, 40, 0.05
Z0 = np.zeros(NX)
TARGET = np.array([3.0, 2.0, 0.0, 0.0, 0.0, 0.0])

def dynamics(z, u):
    T = G + u[0]
    return ca.vertcat(z[3], z[4], z[5],
                      -T * ca.sin(z[2]),
                      T * ca.cos(z[2]) - G,
                      J_INV * u[1])

# === REFERENCE PIECES ===
# One RK4 step and its linearization, used for the adjoint recursion:
def _build():
    z = ca.MX.sym("z", NX)
    u = ca.MX.sym("u", NU)
    k1 = dynamics(z, u)
    k2 = dynamics(z + 0.5 * DT * k1, u)
    k3 = dynamics(z + 0.5 * DT * k2, u)
    k4 = dynamics(z + DT * k3, u)
    zn = z + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    step = ca.Function("step", [z, u], [zn])
    lin = ca.Function("lin", [z, u], [ca.jacobian(zn, z), ca.jacobian(zn, u)])

    # Same rollout, for the IPOPT reference solve:
    U = ca.MX.sym("U", N * NU)
    Zc = step.mapaccum("roll", N)(ca.DM(Z0), ca.reshape(U, NU, N))
    gend = Zc[:, -1] - ca.DM(TARGET)
    return lin, U, gend

LIN, U_SYM, G_END = _build()

# Direct transcription of the same problem: states and controls are both
# decision variables and the dynamics are per-node equality constraints, so
# IPOPT returns one multiplier per node -- the interior costate itself.
def ipopt_transcription():
    z = ca.MX.sym("zs", NX)
    u = ca.MX.sym("us", NU)
    k1 = dynamics(z, u)
    k2 = dynamics(z + 0.5 * DT * k1, u)
    k3 = dynamics(z + 0.5 * DT * k2, u)
    k4 = dynamics(z + DT * k3, u)
    step = ca.Function("st", [z, u], [z + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

    Zv = [ca.MX.sym(f"z{k}", NX) for k in range(N + 1)]
    Uv = [ca.MX.sym(f"u{k}", NU) for k in range(N)]

    # Initial condition, then one defect per interval, then the target:
    gc = [Zv[0] - ca.DM(Z0)]
    for k in range(N):
        gc.append(Zv[k + 1] - step(Zv[k], Uv[k]))
    gc.append(Zv[N] - ca.DM(TARGET))

    X = ca.vertcat(*(Zv + Uv))
    Gc = ca.vertcat(*gc)
    f = sum(ca.dot(Uv[k], Uv[k]) for k in range(N))
    t0 = time.perf_counter()
    S = ca.nlpsol("tr", "ipopt", dict(x=X, f=f, g=Gc),
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-12),
                       print_time=0))
    t_build = time.perf_counter() - t0
    ng = Gc.shape[0]
    t0 = time.perf_counter()
    sol = S(x0=np.zeros(X.shape[0]), lbg=np.zeros(ng), ubg=np.zeros(ng))
    t_solve = time.perf_counter() - t0
    xv = np.array(sol["x"]).flatten()
    mg = np.array(sol["lam_g"]).flatten()

    # Multiplier layout: initial block, N dynamics blocks, terminal block.
    # The dynamics multipliers are the costates at nodes 1..N:
    Zt = xv[:(N + 1) * NX].reshape(N + 1, NX)
    Ut = xv[(N + 1) * NX:].reshape(N, NU)
    nu_dyn = mg[NX:NX + N * NX].reshape(N, NX)
    return (Zt, Ut, nu_dyn, S.stats()["success"], t_build, t_solve,
            S.stats()["iter_count"])

# Minimum-effort reference solve, multipliers taken from the NLP:
def ipopt_reference():
    nlp = dict(x=U_SYM, f=ca.dot(U_SYM, U_SYM), g=G_END)
    t0 = time.perf_counter()
    S = ca.nlpsol("ref", "ipopt", nlp,
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-12),
                       print_time=0))
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol = S(x0=np.zeros(N * NU), lbg=np.zeros(NX), ubg=np.zeros(NX))
    t_solve = time.perf_counter() - t0
    return (np.array(sol["x"]).flatten(),
            np.array(sol["lam_g"]).flatten(),
            S.stats()["success"], t_build, t_solve,
            S.stats()["iter_count"])

# === RUN ===
if __name__ == "__main__":

    # GRACE nominal, unconstrained so the PMP conditions are clean:
    t0 = time.perf_counter()
    system = grace.build(dynamics, nx=NX, nu=NU, N=N, z0=Z0, dt=DT,
                         job="costate")
    engine = grace.GRACE(system)
    t_build_gr = time.perf_counter() - t0

    t0 = time.perf_counter()
    U = np.asarray(engine.shooting.lambda_shoot(TARGET)).flatten()
    t_solve_gr = time.perf_counter() - t0

    Z = np.asarray(system.rollout(U))
    g, Co = system.endpoint_jac(U)

    # GRACE terminal costate from the Gramian solve (R = I here):
    t0 = time.perf_counter()
    W = Co @ Co.T
    lam_T = np.linalg.solve(W + 1e-14 * np.eye(NX), Co @ (2.0 * U))
    t_costate_gr = time.perf_counter() - t0

    # IPOPT references: single shooting for the terminal dual, transcription
    # for the interior costate sequence:
    U_ip, mu_ip, ok, t_b_ss, t_s_ss, it_ss = ipopt_reference()
    Z_tr, U_tr, nu_tr, ok_tr, t_b_tr, t_s_tr, it_tr = ipopt_transcription()

    # === CHECK 1: costate vs NLP multiplier (expect mu = -lam) ===
    d_dual = np.linalg.norm(lam_T + mu_ip) / max(np.linalg.norm(lam_T), 1e-15)
    d_prim = np.linalg.norm(U - U_ip) / max(np.linalg.norm(U_ip), 1e-15)

    # === CHECK 2 and 3: adjoint recursion and Hamiltonian stationarity ===
    lam = np.zeros((N + 1, NX))
    lam[N] = lam_T
    pmp = np.zeros(N)
    for k in range(N - 1, -1, -1):
        A, B = LIN(Z[k], U[k * NU:(k + 1) * NU])
        A = np.array(A)
        B = np.array(B)
        lam[k] = A.T @ lam[k + 1]
        pmp[k] = np.linalg.norm(2.0 * U[k * NU:(k + 1) * NU]
                                - B.T @ lam[k + 1])
    u_scale = max(np.abs(2.0 * U).max(), 1e-15)

    # === REPORT ===
    print(f"IPOPT converged      : {ok}")
    print(f"endpoint error       : {np.linalg.norm(g - system.target(TARGET)):.3e}")
    print(f"control effort       : GRACE {U @ U:.6f}   IPOPT {U_ip @ U_ip:.6f}")
    print(f"\nprimal agreement     : ||U - U_ipopt|| / ||U_ipopt|| = {d_prim:.3e}")
    print(f"dual   agreement     : ||lam + mu_ipopt|| / ||lam||   = {d_dual:.3e}")
    print(f"\nGRACE terminal costate: {np.array2string(lam_T, precision=4)}")
    print(f"IPOPT multiplier      : {np.array2string(mu_ip, precision=4)}")
    print(f"\nstationarity 2U - Co^T lam : "
          f"{np.linalg.norm(2 * U - Co.T @ lam_T) / np.linalg.norm(2 * U):.3e}")
    print(f"PMP residual  max {pmp.max():.3e}   "
          f"relative {pmp.max() / u_scale:.3e}")

    # === CHECK 4: interior costates against the transcription multipliers ===
    # The dynamics multipliers are the costates at nodes 1..N, same sign:
    lam_scale = max(np.abs(lam[1:]).max(), 1e-15)
    d_int = np.abs(lam[1:] - nu_tr).max() / lam_scale
    d_state = np.abs(Z - Z_tr).max() / max(np.abs(Z_tr).max(), 1e-15)
    print(f"\ntranscription converged : {ok_tr}")
    print(f"cost (transcription)    : {float((U_tr ** 2).sum()):.6f}")
    print(f"state agreement         : max rel diff {d_state:.3e}")
    print(f"interior costate agree  : max rel diff {d_int:.3e}")

    # === PLOTS ===
    t = np.arange(N + 1) * DT
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # Every costate the three methods produce, on one axis.
    #
    #   lines   GRACE: the terminal costate comes straight out of the Gramian
    #           solve, and the interior from the adjoint recursion run back
    #           through the linearizations
    #   dots    IPOPT transcription: the multipliers on the per node dynamics
    #           defects are the interior costates, same sign, no recursion
    #   crosses IPOPT single shooting: only the terminal one is available, as
    #           the multiplier on the endpoint constraint, with mu = -lam
    names = ["x", "y", "th", "vx", "vy", "om"]
    for j in range(NX):
        ax[0].plot(t, lam[:, j], "-", lw=1.6, color=f"C{j}",
                   label=f"$\\lambda_{{{names[j]}}}$")
        ax[0].plot(t[1:], nu_tr[:, j], "o", ms=3.5, color=f"C{j}", alpha=0.55)
        ax[0].plot(t[-1], -mu_ip[j], "x", ms=9, mew=2.0, color=f"C{j}")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("costate")
    ax[0].set_title("GRACE (lines), transcription duals (dots),\n"
                    f"single shooting terminal (x) -- max rel diff {d_int:.1e}")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)

    # Control against its PMP prediction from the costate:
    Un = U.reshape(N, NU)
    ax[1].plot(t[:-1], Un[:, 0], "-", color="steelblue", lw=1.8,
               label="thrust dev (solved)")
    ax[1].plot(t[:-1], Un[:, 1], "-", color="darkorange", lw=1.8,
               label="torque (solved)")
    Upmp = np.zeros((N, NU))
    for k in range(N):
        _, B = LIN(Z[k], U[k * NU:(k + 1) * NU])
        Upmp[k] = 0.5 * np.array(B).T @ lam[k + 1]
    ax[1].plot(t[:-1], Upmp[:, 0], "--", color="k", lw=1.0,
               label="$B^T\\lambda/2$ (PMP)")
    ax[1].plot(t[:-1], Upmp[:, 1], "--", color="k", lw=1.0)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("control")
    ax[1].set_title("Control vs PMP prediction")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # Costate agreement at the terminal node:
    idx = np.arange(NX)
    ax[2].bar(idx - 0.2, lam_T, width=0.4, color="steelblue",
              label="GRACE $\\lambda_N$")
    ax[2].bar(idx + 0.2, -mu_ip, width=0.4, color="0.5",
              label="IPOPT $-\\mu$")
    ax[2].set_xticks(idx)
    ax[2].set_xticklabels(names)
    ax[2].set_ylabel("multiplier")
    ax[2].set_title(f"Terminal costate (rel. diff {d_dual:.1e})")
    ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("figures/tests/costate_comparison.png", dpi=140, bbox_inches="tight")
    # === TIMING ===
    # Costate construction is one nx x nx solve on top of the shoot, so it is
    # reported separately from the trajectory solve:
    print("\n=== timing ===")
    print(f"{'method':<26}{'build s':>10}{'solve s':>10}"
          f"{'iters':>8}{'vars':>7}")
    print(f"{'GRACE shoot':<26}{t_build_gr:>10.4f}{t_solve_gr:>10.4f}"
          f"{'-':>8}{N * NU:>7}")
    print(f"{'  + costate':<26}{'':>10}{t_costate_gr:>10.4f}{'':>8}{'':>7}")
    print(f"{'IPOPT single shooting':<26}{t_b_ss:>10.4f}{t_s_ss:>10.4f}"
          f"{it_ss:>8}{N * NU:>7}")
    print(f"{'IPOPT transcription':<26}{t_b_tr:>10.4f}{t_s_tr:>10.4f}"
          f"{it_tr:>8}{(N + 1) * NX + N * NU:>7}")
    tg = t_solve_gr + t_costate_gr
    print(f"\nsolve speedup vs single shooting : "
          f"{t_s_ss / max(tg, 1e-12):.2f}x")
    print(f"solve speedup vs transcription   : "
          f"{t_s_tr / max(tg, 1e-12):.2f}x")
    print(f"total speedup vs single shooting : "
          f"{(t_b_ss + t_s_ss) / max(t_build_gr + tg, 1e-12):.2f}x")

    print("\nsaved figures/tests/costate_comparison.png")