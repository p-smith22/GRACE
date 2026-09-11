# ============================================================================
# example_compare.py -- value field vs Gramian deformation operator
# ============================================================================
# Two representations of the same reachable surfaces.
#
#   FIELD      sublevel sets of V(x), the minimum effort to reach x. Assumes
#              nothing about the shape of the set, so folded, lobed and banded
#              level sets all survive. Costs one solve per grid cell.
#
#   OPERATOR   the Gramian ellipsoid deformed radially. A ray meets the set in
#              a union of intervals; the dominant one is compressed across the
#              budget family as log rho = mu(theta) + a_c phi(theta) on its
#              angular support, and the transient ones are carried explicitly.
#
# Two numbers are reported for the operator and they mean different things.
# At n_modes = 4 with five budgets the modal step is LOSSLESS, so that column
# measures the representation: whether an ellipsoid plus radial intervals can
# describe the set at all. At n_modes = 1 it measures the compression: whether
# the budget family collapses to one profile plus a scalar. The first holds on
# every plant tried; the second degrades as the set folds.
# ============================================================================
import time
import numpy as np
import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grace
from grace.reachability import analysis as ra

NG = 320
NG_COARSE = 56
N_STEP = 8
SUBSTEPS = 24
N_DIR = 720
KMAX = 6
FTOL = 1e-3
FRAC = [0.2, 0.4, 0.6, 0.8, 1.0]

# === PLANTS ===
def integrator():

    def f(z, u):
        return ca.vertcat(z[1], u[0])

    return dict(name="double integrator", f=f, nx=2, T=3.0, budget=6.0,
                ext=[-6.0, 6.0, -3.6, 3.6],
                labels=("endpoint position", "endpoint rate"))

def unicycle():

    def f(z, u):
        return ca.vertcat(ca.cos(z[2]), ca.sin(z[2]), u[0])

    return dict(name="unicycle", f=f, nx=3, T=3.0, budget=8.0,
                ext=[-1.5, 3.2, -2.9, 2.9],
                labels=("endpoint x", "endpoint y"))

def duffing():

    def f(z, u):
        return ca.vertcat(z[1], z[0] - z[0] ** 3 - 0.1 * z[1] + u[0])

    return dict(name="Duffing", f=f, nx=2, T=4.0, budget=2.0,
                ext=[-2.2, 2.2, -2.3, 2.3],
                labels=("endpoint position", "endpoint rate"))

def van_der_pol():

    def f(z, u):
        return ca.vertcat(z[1], (1 - z[0] ** 2) * z[1] - z[0] + u[0])

    return dict(name="Van der Pol", f=f, nx=2, T=8.0, budget=2.0,
                ext=[-2.6, 2.6, -4.3, 4.3],
                labels=("endpoint position", "endpoint rate"))

def pendulum():

    def f(z, u):
        return ca.vertcat(z[1], -ca.sin(z[0]) + u[0])

    return dict(name="pendulum", f=f, nx=2, T=10.0, budget=4.0,
                ext=[-45.0, 45.0, -8.0, 8.0],
                labels=("endpoint angle", "endpoint rate"))

PLANTS = [integrator, van_der_pol, duffing, unicycle, pendulum]

def iou(A, B):
    return (A & B).sum() / max((A | B).sum(), 1)

# Mean intervals per ray, which grades how folded the set is:
def fold_count(intervals):
    iv = intervals[-1]

    return float(np.mean((iv[:, :, 1] > iv[:, :, 0]).sum(axis=1)))

# === MAIN ===
if __name__ == "__main__":

    n = len(PLANTS)
    fig1, ax1 = plt.subplots(1, n, figsize=(4.6 * n, 4.6))
    ax1 = np.atleast_1d(ax1)
    rows = []

    print(f"{'plant':18s} {'field s':>8s} {'solves':>7s} {'op s':>6s} "
          f"{'folds':>6s} | {'IoU k=4':>8s} {'IoU k=1':>8s}")

    for ax, make in zip(ax1, PLANTS):
        p = make()
        ext = p["ext"]
        b = p["budget"]
        budgets = [b * f for f in FRAC]
        U0 = np.zeros(N_STEP)

        system = grace.build(p["f"], p["nx"], 1, N_STEP, np.zeros(p["nx"]),
                             p["T"] / N_STEP, target_idx=[0, 1],
                             substeps=SUBSTEPS)
        engine = grace.GRACE(system)

        # --- FIELD ---
        t0 = time.time()
        V, xs, ys, ctrl, n_solve, n_fail = ra.value_field(
            engine, system, U0, ext, NG_COARSE, b, ftol=FTOL,
            budgets_near=budgets)
        t_field = time.time() - t0
        masks = [ra.sublevel(V, xs, ys, ext, g, NG) for g in budgets]

        # --- OPERATOR ---
        e0 = np.asarray(system.endpoint(U0)).ravel()[:2]
        P = ra._inv_psd(ra.gramian(system, U0, cost=None))
        th = np.linspace(0.0, 2.0 * np.pi, N_DIR, endpoint=False)

        t0 = time.time()
        IV = [ra.ray_intervals(M, ext, e0, th, kmax=KMAX) for M in masks]
        acc = {}
        best = None
        for k in (4, 1):
            fit = ra.fit_deformation(th, IV, budgets, P, e0, n_modes=k)
            acc[k] = float(np.mean(
                [iou(masks[j], ra.deformed_mask(fit, budgets[j], ext, NG))
                 for j in range(len(budgets))]))
            if k == 4:
                best = fit
        t_op = time.time() - t0

        folds = fold_count(IV)
        rows.append((p["name"], t_field, n_solve, t_op, acc[4], acc[1], folds))
        print(f"{p['name']:18s} {t_field:8.1f} {n_solve:7d} {t_op:6.1f} "
              f"{folds:6.2f} | {acc[4]:8.3f} {acc[1]:8.3f}")

        # --- SHAPE ---
        shades = plt.cm.viridis(np.linspace(0.85, 0.2, len(budgets)))
        for M, col in zip(masks[::-1], shades):
            over = np.zeros(M.shape + (4,))
            over[M] = col
            ax.imshow(np.transpose(over, (1, 0, 2)), origin="lower", extent=ext)

        gx = np.linspace(ext[0], ext[1], NG)
        gy = np.linspace(ext[2], ext[3], NG)
        for g in budgets:
            R = ra.deformed_mask(best, g, ext, NG)
            ax.contour(gx, gy, R.T, [0.5], colors="k", linewidths=1.2)

        ax.set_title(f"{p['name']}\nIoU {acc[4]:.3f}   folds/ray {folds:.2f}",
                     fontsize=9)
        ax.set_xlabel(p["labels"][0], fontsize=8)
        ax.set_ylabel(p["labels"][1], fontsize=8)
        ax.tick_params(labelsize=7)

    fig1.suptitle("Value field (shaded) against the Gramian deformation "
                  "operator (outlined)", fontsize=12)
    fig1.tight_layout()
    fig1.savefig("compare_shape.png", dpi=110, bbox_inches="tight")

    # === COST AND ACCURACY ===
    fig2, (a, c) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    names = [r[0] for r in rows]
    x = np.arange(len(names))
    a.bar(x - 0.2, [r[1] for r in rows], 0.4, label="field solve", color="0.35")
    a.bar(x + 0.2, [r[3] for r in rows], 0.4, label="operator fit",
          color="tab:green")
    a.set_xticks(x)
    a.set_xticklabels(names, fontsize=7, rotation=20)
    a.set_ylabel("seconds")
    a.set_title("construction cost", fontsize=10)
    a.legend(fontsize=8)

    c.plot(x, [r[4] for r in rows], "ko-", label="representation (k=4)")
    c.plot(x, [r[5] for r in rows], "s--", color="tab:red",
           label="compression (k=1)")
    c.set_xticks(x)
    c.set_xticklabels(names, fontsize=7, rotation=20)
    c.set_ylim(0.0, 1.05)
    c.set_ylabel("IoU against the field")
    c.set_title("representation holds everywhere, compression degrades with "
                "folding", fontsize=9)
    c.legend(fontsize=8)

    fig2.tight_layout()
    fig2.savefig("compare_cost.png", dpi=110, bbox_inches="tight")
    print("\nfigures: compare_shape.png, compare_cost.png")