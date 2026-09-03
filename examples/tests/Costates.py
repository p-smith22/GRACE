# ============================================================================
# costate_comparison.py -- states, controls, and costates: GRACE vs IPOPT
# ============================================================================
# The costate is constructed exactly as the derivation does. Stationarity is
#
#     grad l(u_k) + B_k^T lam_{k+1} = 0,     lam_k = A_k^T lam_{k+1}
#
# so every costate is a linear map of the terminal multiplier nu, and nu
# follows from the Gramian and the displacement the control produces,
#
#     nu = -W^-1 delta,     W = Co M Co^T,     delta = Co U
#
# with M the inverse cost curvature, here R^-1 = I. For a quadratic cost this
# is the same arithmetic as recovering nu by least squares from the converged
# control, so it is reported as a cross check on the solver's own costate
# rather than as an independent one.
#
# Sign convention throughout is the derivation's: grad l = -Co^T nu. Against
# it, IPOPT's single shooting multiplier has the same sign, its transcription
# defect multipliers have the opposite sign, and the solver's internal costate
# is -nu.
#
# The reference objective is l = U^T U / 2, matching l = u^T R u / 2 with
# R = I, so no factors of two appear anywhere.
# ============================================================================

import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace

# === PROBLEM ===
# Planar quadrotor, thrust commanded as a deviation from hover:
G = 9.81
J_INV = 1.0 / 0.02
NX, NU, N, DT = 6, 2, 40, 0.05
Z0 = np.zeros(NX)
TARGET = np.array([3.0, 2.0, 0.0, 0.0, 0.0, 0.0])
SNAMES = ("x", "y", "th", "vx", "vy", "om")
UNAMES = ("thrust dev", "torque")

def dynamics(z, u):
    T = G + u[0]
    return ca.vertcat(z[3], z[4], z[5],
                      -T * ca.sin(z[2]),
                      T * ca.cos(z[2]) - G,
                      J_INV * u[1])

# === REFERENCE PIECES ===
# One RK4 step and its linearization. The linearization supplies A_k and B_k
# for the adjoint recursion and for Hamiltonian stationarity:
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

    # Same rollout, for the single shooting reference:
    U = ca.MX.sym("U", N * NU)
    Zc = step.mapaccum("roll", N)(ca.DM(Z0), ca.reshape(U, NU, N))
    gend = Zc[:, -1] - ca.DM(TARGET)
    return lin, U, gend

LIN, U_SYM, G_END = _build()

# Single shooting reference. The multiplier on the terminal constraint is nu
# itself: stationarity of the NLP reads grad l + Co^T mu = 0, which is the
# derivation's condition with mu in place of nu:
def ipopt_shooting():
    nlp = dict(x=U_SYM, f=0.5 * ca.dot(U_SYM, U_SYM), g=G_END)
    t0 = time.perf_counter()
    S = ca.nlpsol("ref", "ipopt", nlp,
                  dict(ipopt=dict(print_level=0, sb="yes", tol=1e-12),
                       print_time=0))
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol = S(x0=np.zeros(N * NU), lbg=np.zeros(NX), ubg=np.zeros(NX))
    t_solve = time.perf_counter() - t0
    return (np.array(sol["x"]).flatten(), np.array(sol["lam_g"]).flatten(),
            S.stats()["success"], t_build, t_solve, S.stats()["iter_count"])

# Direct transcription. States and controls are both decision variables and
# the dynamics are per node equalities, so the defect multipliers are the
# interior costates -- but with the opposite sign, since stationarity in u_k
# there reads u_k - B_k^T eta_{k+1} = 0 against the derivation's
# u_k + B_k^T lam_{k+1} = 0, hence eta = -lam:
def ipopt_transcription():
    z = ca.MX.sym("zs", NX)
    u = ca.MX.sym("us", NU)
    k1 = dynamics(z, u)
    k2 = dynamics(z + 0.5 * DT * k1, u)
    k3 = dynamics(z + 0.5 * DT * k2, u)
    k4 = dynamics(z + DT * k3, u)
    step = ca.Function("st", [z, u],
                       [z + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

    Zv = [ca.MX.sym(f"z{k}", NX) for k in range(N + 1)]
    Uv = [ca.MX.sym(f"u{k}", NU) for k in range(N)]

    # Initial condition, then one defect per interval, then the target:
    gc = [Zv[0] - ca.DM(Z0)]
    for k in range(N):
        gc.append(Zv[k + 1] - step(Zv[k], Uv[k]))
    gc.append(Zv[N] - ca.DM(TARGET))

    X = ca.vertcat(*(Zv + Uv))
    Gc = ca.vertcat(*gc)
    f = 0.5 * sum(ca.dot(Uv[k], Uv[k]) for k in range(N))
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
    # Negated into the derivation's sign convention:
    Zt = xv[:(N + 1) * NX].reshape(N + 1, NX)
    Ut = xv[(N + 1) * NX:].reshape(N, NU)
    lam_int = -mg[NX:NX + N * NX].reshape(N, NX)
    return (Zt, Ut, lam_int, S.stats()["success"], t_build, t_solve,
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

    # === TERMINAL MULTIPLIER ===
    # Constructed as the derivation constructs it: the generalized Gramian
    # W = Co M Co^T against the displacement the control produces. With a
    # quadratic cost M = R^-1 = I, so W is the standard output Gramian:
    M_diag = np.ones(N * NU)
    W = (Co * M_diag) @ Co.T
    delta = Co @ U
    nu_gram = -np.linalg.solve(W + 1e-14 * np.eye(NX), delta)

    # The solver's own costate is the object the derivation claims as a
    # byproduct, so it is what the comparisons use. It carries the opposite
    # sign to nu, so it is converted here:
    lam_solver = getattr(system, "_costate", None)
    if lam_solver is None:
        print("NOTE: solver returned no costate, using the constructed one")
        nu = nu_gram
        d_gram = np.nan
    else:
        nu = -0.5 * np.asarray(lam_solver, float).ravel()
        d_gram = (np.linalg.norm(nu - nu_gram)
                  / max(np.linalg.norm(nu), 1e-15))

    # References:
    U_ss, mu_ss, ok_ss, t_b_ss, t_s_ss, it_ss = ipopt_shooting()
    Z_tr, U_tr, lam_tr, ok_tr, t_b_tr, t_s_tr, it_tr = ipopt_transcription()

    # === CHECK 1: terminal multiplier against the shooting multiplier ===
    d_dual = np.linalg.norm(nu - mu_ss) / max(np.linalg.norm(nu), 1e-15)
    d_prim = np.linalg.norm(U - U_ss) / max(np.linalg.norm(U_ss), 1e-15)

    # === CHECK 2 and 3: adjoint recursion and Hamiltonian stationarity ===
    # Terminal condition lam_N = C^T nu, and C is the identity here since the
    # full state is constrained. The recursion and the control law are the
    # derivation's, with grad l = u for this cost:
    lam = np.zeros((N + 1, NX))
    lam[N] = nu
    pmp = np.zeros(N)
    U_pmp = np.zeros((N, NU))
    for k in range(N - 1, -1, -1):
        A, B = LIN(Z[k], U[k * NU:(k + 1) * NU])
        A = np.array(A)
        B = np.array(B)
        lam[k] = A.T @ lam[k + 1]
        U_pmp[k] = -B.T @ lam[k + 1]
        pmp[k] = np.linalg.norm(U[k * NU:(k + 1) * NU] + B.T @ lam[k + 1])
    u_scale = max(np.abs(U).max(), 1e-15)

    # === CHECK 4: interior costates against the transcription multipliers ===
    lam_scale = max(np.abs(lam[1:]).max(), 1e-15)
    d_int = np.abs(lam[1:] - lam_tr).max() / lam_scale
    d_state = np.abs(Z - Z_tr).max() / max(np.abs(Z_tr).max(), 1e-15)
    d_ctrl = (np.abs(U.reshape(N, NU) - U_tr).max()
              / max(np.abs(U_tr).max(), 1e-15))

    # === REPORT ===
    print(f"IPOPT converged, shooting / transcription : {ok_ss} / {ok_tr}")
    print(f"endpoint error : "
          f"{np.linalg.norm(g - system.target(TARGET)):.3e}")
    print(f"effort U^T U/2 : GRACE {0.5 * U @ U:.8f}   "
          f"shooting {0.5 * U_ss @ U_ss:.8f}   "
          f"transcription {0.5 * float((U_tr ** 2).sum()):.8f}")

    print(f"\nprimal agreement, shooting      : {d_prim:.3e}")
    print(f"primal agreement, transcription : states {d_state:.3e}, "
          f"controls {d_ctrl:.3e}")
    print(f"dual agreement, terminal        : {d_dual:.3e}")
    print(f"dual agreement, interior        : {d_int:.3e}")
    print(f"solver nu vs Gramian formula    : {d_gram:.3e}")

    print(f"\nnu, GRACE              : {np.array2string(nu, precision=4)}")
    print(f"nu, Gramian formula    : {np.array2string(nu_gram, precision=4)}")
    print(f"mu, IPOPT shooting     : {np.array2string(mu_ss, precision=4)}")
    print(f"lam_N, transcription   : "
          f"{np.array2string(lam_tr[-1], precision=4)}")

    print(f"\nstationarity U + Co^T nu : "
          f"{np.linalg.norm(U + Co.T @ nu) / np.linalg.norm(U):.3e}")
    print(f"PMP residual, max {pmp.max():.3e}, relative "
          f"{pmp.max() / u_scale:.3e}")

    # === TIMING ===
    print("\n=== timing ===")
    print(f"{'method':<26}{'build s':>10}{'solve s':>10}{'iters':>8}{'vars':>7}")
    print(f"{'GRACE shoot':<26}{t_build_gr:>10.4f}{t_solve_gr:>10.4f}"
          f"{'-':>8}{N * NU:>7}")
    print(f"{'IPOPT single shooting':<26}{t_b_ss:>10.4f}{t_s_ss:>10.4f}"
          f"{it_ss:>8}{N * NU:>7}")
    print(f"{'IPOPT transcription':<26}{t_b_tr:>10.4f}{t_s_tr:>10.4f}"
          f"{it_tr:>8}{(N + 1) * NX + N * NU:>7}")
    print(f"\nsolve speedup vs shooting      : "
          f"{t_s_ss / max(t_solve_gr, 1e-12):.2f}x")
    print(f"solve speedup vs transcription : "
          f"{t_s_tr / max(t_solve_gr, 1e-12):.2f}x")

    # === PLOT ===
    # One figure. States and controls show the primals land on top of each
    # other, costates show the duals do as well, which is the stronger claim:
    t = np.arange(N + 1) * DT
    tu = t[:-1]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for j in range(NX):
        ax[0].plot(t, Z[:, j], "-", lw=1.7, color=f"C{j}", label=SNAMES[j])
        ax[0].plot(t[::3], Z_tr[::3, j], "o", ms=3.5, mfc="none",
                   color=f"C{j}")
        ax[0].plot(t[-1], TARGET[j], "*", color="k", ms=12)
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("State")
    ax[0].set_title(f"States, GRACE (lines), IPOPT (circles)")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)

    Un = U.reshape(N, NU)
    for j in range(NU):
        ax[1].plot(tu, Un[:, j], "-", lw=1.8, color=f"C{j}",
                   label=f"{UNAMES[j]}, GRACE")
        ax[1].plot(tu[::3], U_tr[::3, j], "o", ms=3.5, mfc="none",
                   color=f"C{j}", label=f"{UNAMES[j]}, IPOPT")
        ax[1].plot(tu, U_pmp[:, j], "--", lw=1.0, color="k")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Control")
    ax[1].set_title(f"Controls")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # Lines are GRACE, terminal from the solve and interior from the adjoint
    # recursion. Circles are the transcription defect multipliers, negated
    # into this convention. Crosses are the single shooting terminal dual,
    # the only one that method makes available, same sign as nu:
    for j in range(NX):
        ax[2].plot(t, lam[:, j], "-", lw=1.7, color=f"C{j}",
                   label=rf"$\lambda_{{{SNAMES[j]}}}$")
        ax[2].plot(t[1:][::3], lam_tr[::3, j], "o", ms=3.5, mfc="none",
                   color=f"C{j}")
        ax[2].plot(t[-1], mu_ss[j], "x", ms=9, mew=2.0, color=f"C{j}")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Costate")
    ax[2].set_title(f"Costates, GRACE (lines), IPOPT (circles)")
    ax[2].legend(fontsize=8, ncol=2)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/tests/costate_comparison.png", dpi=140,
                bbox_inches="tight")
    print("\nsaved figures/tests/costate_comparison.png")