# Import packages:
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace.codesign.codesign as cd

# Define norms to be analyzed:
NORMS = ("l1", "l2", "cheby")
COLORS = dict(l1="crimson", l2="darkorange", cheby="steelblue")
MARKERS = dict(l1="o", l2="s", cheby="^")

# Planar point mass under gravity. z = [x, y, vx, vy]:
def dynamics(z, u, p):
    return ca.vertcat(z[2], z[3],
                      p * u[0],
                      p * u[1] - 9.81)

# Design origin and bounds:
p0 = 1.5
p_bounds = (0.5, 2.0)

# Define trajectory and sweep:
nx, nu, N, dt = 4, 2, 100, 0.05
z0 = np.zeros(nx)
target = np.array([5.0, 5.0, 0.0, 0.0])
weights = np.linspace(0.0, 1.0, 21)

# Define step parameters:
P_STEP = 1.20
STEP_WIDTH = 0.03
STEP_HEIGHT = 900.0

# Define mass function:
def mass(p, exp_fn):

    # Smooth cubic term:
    smooth = 120.0 * p ** 3

    # Step, which makes the front non-convex.
    # Represents where the design needs the next motor/battery for further improvement:
    step = STEP_HEIGHT / (1.0 + exp_fn(-(p - P_STEP) / STEP_WIDTH))

    # Return the combined mass:
    return smooth + step

# Define NumPy version:
def objective(p):
    return float(mass(float(p), np.exp))

# Define CasADi version:
def objective_ca(p):
    return mass(p, ca.exp)

# IPOPT build:
def build_ipopt(nrm, nz):

    # Define symbolic variables:
    U = ca.MX.sym("U", N * nu)
    p = ca.MX.sym("p", 1)
    t = ca.MX.sym("t", 1)
    w1 = ca.MX.sym("w1", 1)
    w2 = ca.MX.sym("w2", 1)

    # Propagate trajectory:
    z = ca.DM(z0)
    for k in range(N):
        u = U[k * nu:(k + 1) * nu]
        k1 = dynamics(z, u, p)
        k2 = dynamics(z + 0.5 * dt * k1, u, p)
        k3 = dynamics(z + 0.5 * dt * k2, u, p)
        k4 = dynamics(z + dt * k3, u, p)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Scalarize cost and design objectives:
    Chat = (ca.dot(U, U) - nz["C_id"]) / nz["C_rng"]
    Dhat = (objective_ca(p) - nz["D_id"]) / nz["D_rng"]

    # Compute endpoint error:
    g_end = z - ca.DM(target)

    # If L-inf (cheby) norm:
    if nrm == "cheby":

        # Define objective and constraints:
        x = ca.vertcat(U, p, t)
        f = t + 1e-3 * Chat
        g = ca.vertcat(g_end, t - w1 * Chat, t - w2 * Dhat)

        # Set bounds:
        lbx = np.r_[np.full(N * nu, -np.inf), p_bounds[0], 0.0]
        ubx = np.r_[np.full(N * nu, np.inf), p_bounds[1], np.inf]
        lbg = np.r_[np.zeros(nx), np.zeros(2)]
        ubg = np.r_[np.zeros(nx), np.full(2, np.inf)]

    # If L1 or L2 norm:
    else:

        # Define objective and constraints:
        x = ca.vertcat(U, p)
        f = (w1 * Chat + w2 * Dhat if nrm == "l1"
             else (w1 * Chat) ** 2 + (w2 * Dhat) ** 2) + 1e-3 * Chat
        g = g_end

        # Set bounds:
        lbx = np.r_[np.full(N * nu, -np.inf), p_bounds[0]]
        ubx = np.r_[np.full(N * nu, np.inf), p_bounds[1]]
        lbg = ubg = np.zeros(nx)

    # Build solver:
    solver = ca.nlpsol("mono", "ipopt",
                       dict(x=x, p=ca.vertcat(w1, w2), f=f, g=g),
                       dict(ipopt=dict(print_level=0, sb="yes", tol=1e-8),
                            print_time=0))

    # Return solver and bounds:
    return solver, lbx, ubx, lbg, ubg

# Main run loop:
if __name__ == "__main__":

    # === SOLVE ===
    # Fetch number of weights and initialize results:
    nw = len(weights)
    res = {}

    # Loop through norms:
    for nrm in NORMS:

        # GRACE front:
        t0 = time.perf_counter()
        _, _, pareto, sweep = cd.codesign(
            dynamics, nx, nu, N, z0, dt, target, "thrust_authority", objective,
            p0, p_bounds, weights=weights, norm=nrm, n_anchor=9, plot=False)
        t_gr = time.perf_counter() - t0
        pareto = sorted(pareto, key=lambda f: f["weight"])

        # Ideal point from the same sweep, shared with the baseline:
        Cg = np.array([s_["cost"] for s_ in sweep])
        Dg = np.array([s_["objective"] for s_ in sweep])
        C_id = Cg.min() - 0.01 * np.ptp(Cg)
        D_id = Dg.min() - 0.01 * np.ptp(Dg)
        nz = dict(C_id=C_id, D_id=D_id,
                  C_rng=max(Cg[int(np.argmin(Dg))] - C_id, 1e-12),
                  D_rng=max(Dg[int(np.argmin(Cg))] - D_id, 1e-12))

        # IPOPT front on the identical scalarization:
        solver, lbx, ubx, lbg, ubg = build_ipopt(nrm, nz)
        xg = (np.r_[np.zeros(N * nu), p0, 1.0] if nrm == "cheby"
              else np.r_[np.zeros(N * nu), p0])
        mono, t_ip = [], 0.0
        for w in weights:
            t1 = time.perf_counter()
            sol = solver(x0=xg, p=[1.0 - w, w],
                         lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            t_ip += time.perf_counter() - t1
            xg = np.array(sol["x"]).flatten()
            mono.append(dict(weight=float(w), param=float(xg[N * nu]),
                             cost=float(xg[:N * nu] @ xg[:N * nu])))

        # The dominance filter can drop points, so pair the fronts by weight:
        mw = {round(f["weight"], 12): f for f in mono}
        pr = [(a, mw[round(a["weight"], 12)]) for a in pareto
              if round(a["weight"], 12) in mw]
        dp = max(abs(a["param"] - b["param"]) for a, b in pr) if pr else np.nan

        # Package results:
        res[nrm] = dict(pareto=pareto, mono=mono, sweep=sweep,
                        t_gr=t_gr, t_ip=t_ip, dp=dp, n_pareto=len(pareto),
                        pg=np.array([f["param"] for f in pareto]),
                        pi=np.array([f["param"] for f in mono]))

    # === RESULTS ===
    # Find span of design objective:
    D_lo = objective(p_bounds[1])
    D_hi = objective(p_bounds[0])
    D_span = abs(D_hi - D_lo)

    # Find largest jump in design objective, normalized by span:
    def gap_frac(pv):
        Dv = np.sort(np.array([objective(x) for x in pv]))
        return (float(np.diff(Dv).max()) / D_span * 100.0
                if Dv.size > 1 else 0.0)

    # Print results:
    print(f"{'norm':<8}{'GRACE ms':>10}{'IPOPT ms':>10}{'front pts':>11}"
          f"{'GRACE gap %':>13}{'IPOPT gap %':>13}{'max |dp|':>10}")
    for nrm in NORMS:
        d = res[nrm]
        print(f"{nrm:<8}{d['t_gr'] / nw * 1e3:>10.2f}"
              f"{d['t_ip'] / nw * 1e3:>10.2f}{d['n_pareto']:>11}"
              f"{gap_frac(d['pg']):>13.1f}{gap_frac(d['pi']):>13.1f}"
              f"{d['dp']:>10.4f}")

    # === PLOT ===
    # Initialize figures:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # Plot fronts:
    for nrm in NORMS:
        d = res[nrm]
        ax[0].plot([f["objective"] for f in d["pareto"]],
                   [f["cost"] for f in d["pareto"]],
                   MARKERS[nrm], color=COLORS[nrm], ms=7, alpha=0.85,
                   label=f"{nrm} (GRACE)")
        ax[0].plot([objective(x) for x in d["pi"]],
                   [f["cost"] for f in d["mono"]], "x",
                   color=COLORS[nrm], ms=7, mew=1.6)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Design Objective (mass)")
    ax[0].set_ylabel("Control Effort")
    ax[0].set_title("Recovered Fronts (x = IPOPT)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    # Plot converged parameters:
    for nrm in NORMS:
        d = res[nrm]
        ax[1].plot([f["weight"] for f in d["pareto"]], d["pg"],
                   MARKERS[nrm] + "-", color=COLORS[nrm], ms=6, lw=1.3,
                   label=f"{nrm} (GRACE)")
        ax[1].plot([f["weight"] for f in d["mono"]], d["pi"], "x",
                   color=COLORS[nrm], ms=7, mew=1.6)
    ax[1].axhline(P_STEP, color="0.5", ls=":", lw=1.2)
    ax[1].set_xlabel("Weight, w")
    ax[1].set_ylabel("Selected Design, p")
    ax[1].set_ylim(p_bounds)
    ax[1].set_title("Weight to Design (x = IPOPT)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # Plot jump in design objective:
    xpos = np.arange(len(NORMS))
    ng = [gap_frac(res[n]["pg"]) for n in NORMS]
    ni = [gap_frac(res[n]["pi"]) for n in NORMS]
    ax[2].bar(xpos - 0.18, ng, 0.36, color=[COLORS[n] for n in NORMS],
              label="GRACE")
    ax[2].bar(xpos + 0.18, ni, 0.36, color="0.6", label="IPOPT")
    for i in range(len(NORMS)):
        ax[2].text(i - 0.18, ng[i], f"{ng[i]:.0f}%", ha="center", va="bottom",
                   fontsize=9)
        ax[2].text(i + 0.18, ni[i], f"{ni[i]:.0f}%", ha="center", va="bottom",
                   fontsize=9)
    ax[2].set_xticks(xpos)
    ax[2].set_xticklabels(NORMS)
    ax[2].set_ylabel("Largest Jump in Design Objective, %")
    ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.3, axis="y")

    # Layout, save figure:
    fig.tight_layout()
    fig.savefig("figures/tests/codesign.png", dpi=140, bbox_inches="tight")
