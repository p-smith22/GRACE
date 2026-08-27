# ============================================================================
# example_cost.py -- propellant instead of a quadratic control cost
# ============================================================================
# lambda_shoot minimizes ||U||^2 because stationarity makes the control linear
# in the endpoint costate, U = R^-1 Co' lam. Passing cost=(f, dinv, dinv') keeps
# that structure and swaps the map for u_k = dinv((Co' lam)_k), so any convex
# separable cost can be used.
#
# A cold gas thruster burns propellant in proportion to the impulse it delivers,
# plus a loss that grows with throttle:
#
#     f(u) = a |u| + b u^2
#
# The |u| term is what a quadratic cannot imitate. Its marginal cost is a at any
# nonzero throttle, so a costate smaller than a cannot pay to open the valve at
# all and the optimal thrust there is exactly zero. The solution burns, coasts,
# and burns again. A quadratic cost has zero marginal cost at zero thrust, so it
# always finds some tiny correction worth making and trickles thrust across the
# whole horizon.
#
# The difference is qualitative rather than a percentage, which is the point:
# it does not depend on tuning a weight until a gap appears.
# ============================================================================

import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace

# === PLANT ===
# Planar spacecraft with one main engine and a reaction wheel. The engine
# thrusts along the body axis only, so the vehicle cannot translate in a chosen
# direction without first pointing that way. z = [x, y, th, vx, vy, om].
#
# Underactuation is what makes the path depend on the cost. On a vehicle with
# thrusters on both axes the direction of thrust is free at every instant, both
# solutions hold much the same ratio between the axes, and they trace the same
# straight line at different speeds. Here the direction is the attitude, the
# attitude takes time to change, and a solution that wants to burn in bursts
# has to arrange to be pointing the right way when it does.
MASS, IZZ = 12.0, 0.8

def dynamics(z, u):
    th, vx, vy, om = z[2], z[3], z[4], z[5]
    thrust, tau = u[0], u[1]
    return ca.vertcat(vx, vy, om,
                      thrust * ca.cos(th) / MASS,
                      thrust * ca.sin(th) / MASS,
                      tau / IZZ)

nx, nu, N, dt = 6, 2, 40, 0.25
z0 = np.zeros(nx)

# Translate across and arrive at rest. Attitude at the end is left free, since
# where the vehicle ends up pointing is a consequence of how it chose to burn:
target = np.array([4.0, 2.5, 0.0, 0.0, 0.0, 0.0])
TIDX = [0, 1, 3, 4, 5]

# === COSTS ===
# Each cost is what a control history costs, the inverse of its marginal cost,
# and the derivative of that inverse. The second is the only thing the solver
# needs beyond the Jacobian it already forms.
#
# A_PROP is set from the costates the problem actually produces. Below them the
# deadband never activates and the cost is a quadratic in disguise.
A_PROP, B_PROP = 6.0, 1.0

def quadratic():
    # What the solver already minimizes, written in the general form.
    return (lambda U: float(U @ U),
            lambda s: s / 2.0,
            lambda s: np.full_like(s, 0.5))

def propellant():
    # f(u) = a|u| + b u^2. Marginal cost is a + 2b|u| at any nonzero throttle,
    # so the valve stays shut until the costate exceeds a. That threshold is
    # what makes the burn structure discrete rather than smooth.
    def f(U):
        return float(np.sum(A_PROP * np.abs(U) + B_PROP * U ** 2))

    # The valve opens over a narrow band rather than instantly. A hard cutoff
    # leaves the inverse map flat wherever the valve is shut, so the root find
    # has a zero Jacobian there and no direction to move in; the width below is
    # small enough to keep the burn structure and large enough to be solvable.
    W_BAND = 0.04 * A_PROP

    def _dead(s):
        e = np.abs(s) - A_PROP
        return 0.5 * (np.sqrt(e ** 2 + W_BAND ** 2) + e)

    def dinv(s):
        return np.sign(s) * _dead(s) / (2.0 * B_PROP)

    def dinv_prime(s):
        e = np.abs(s) - A_PROP
        return (0.5 * (1.0 + e / np.sqrt(e ** 2 + W_BAND ** 2))
                / (2.0 * B_PROP))

    return (f, dinv, dinv_prime)

# === RUN ===
if __name__ == "__main__":
    system = grace.build_cached(dynamics, nx=nx, nu=nu, N=N, z0=z0, dt=dt,
                                target_idx=TIDX, job="cost_prop")
    shoot = grace.GRACE(system).shooting.lambda_shoot

    f_q = quadratic()[0]
    f_p = propellant()[0]

    runs, timing = {}, {}
    t0 = time.perf_counter()
    U_ref = np.asarray(shoot(target)).flatten()
    timing["built in quadratic"] = time.perf_counter() - t0

    for label, c in (("quadratic", quadratic()),
                     ("propellant", propellant())):
        t0 = time.perf_counter()
        U = np.asarray(shoot(target, cost=c)).flatten()
        timing[f"cost={label}"] = time.perf_counter() - t0
        runs[label] = dict(
            U=U, Z=np.asarray(system.rollout(U)),
            err=float(np.linalg.norm(system.endpoint(U)
                                     - system.target(target))))

    from scipy.optimize import minimize
    t0 = time.perf_counter()
    res = minimize(f_p, runs["propellant"]["U"], method="SLSQP",
                   constraints=[dict(
                       type="eq",
                       fun=lambda v: np.asarray(system.endpoint(v))
                       - system.target(target))],
                   options=dict(maxiter=3000, ftol=1e-12))
    timing["direct NLP on propellant"] = time.perf_counter() - t0

    # Stationarity of what each run returned. At a true optimum the cost
    # gradient is spanned by the endpoint Jacobian, so grad f - Co' lam is zero.
    #
    # The propellant cost is not differentiable at zero thrust: its subgradient
    # there is the whole interval [-a, a], so a|u|' evaluated as a*sign(u) is
    # meaningless wherever the valve is nearly shut. The gradient used below is
    # the one belonging to the smoothed valve the solver actually inverts,
    # which is defined everywhere and agrees with a*sign(u) away from zero.
    def stationarity(U, cost_grad):
        _, Co = system.endpoint_jac(U)
        gr = cost_grad(U)
        lam, *_ = np.linalg.lstsq(Co.T, gr, rcond=None)
        return float(np.linalg.norm(gr - Co.T @ lam)
                     / max(np.linalg.norm(gr), 1e-30))

    grad_q = lambda U: 2.0 * U

    def grad_p(U):
        w = 0.04 * A_PROP
        return A_PROP * U / np.sqrt(U ** 2 + w ** 2) + 2.0 * B_PROP * U

    print(f"{'':<14}{'quadratic':>12}{'propellant':>13}{'valve shut':>12}"
          f"{'stationarity':>14}{'endpoint':>11}")
    for label, d in runs.items():
        U = d["U"]
        shut = np.mean(np.abs(U) < 0.02 * A_PROP / (2.0 * B_PROP)) * 100.0
        st = stationarity(U, grad_q if label == "quadratic" else grad_p)
        print(f"{label:<14}{f_q(U):>12.5g}{f_p(U):>13.5g}{shut:>11.0f}%"
              f"{st:>14.2e}{d['err']:>11.1e}")

    # The same measure applied to the NLP's answer. It is the reference
    # solution, so whatever it scores is what this metric can resolve on this
    # problem -- most of the thrust history sits inside the deadband, where the
    # smoothed gradient turns over sharply and a small error in the control
    # becomes a large one in the gradient.
    print(f"{'direct NLP':<14}{f_q(res.x):>12.5g}{f_p(res.x):>13.5g}"
          f"{np.mean(np.abs(res.x) < 0.02 * A_PROP / (2.0 * B_PROP)) * 100:>11.0f}%"
          f"{stationarity(res.x, grad_p):>14.2e}"
          f"{np.linalg.norm(np.asarray(system.endpoint(res.x)) - system.target(target)):>11.1e}")

    print(f"\n{'run':<26}{'seconds':>10}{'propellant':>13}")
    for k, v in timing.items():
        pv = (f_p(res.x) if k.startswith("direct")
              else f_p(runs["propellant"]["U"]) if "propellant" in k
              else f_p(runs["quadratic"]["U"]) if k.startswith("cost")
              else f_p(U_ref))
        print(f"{k:<26}{v:>10.3f}{pv:>13.5g}")
    print(f"{'agreement, quadratic':<26}"
          f"{np.abs(runs['quadratic']['U'] - U_ref).max():>10.2e}")

    save = 1.0 - f_p(runs["propellant"]["U"]) / f_p(runs["quadratic"]["U"])
    print(f"\npropellant saved by charging for what is burned: "
          f"{save * 100:.1f}%")

    # === PLOT ===
    t = np.arange(N) * dt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = dict(zip(runs, ("0.45", "crimson")))

    for label, d in runs.items():
        Um = d["U"].reshape(N, nu)
        ax[0].plot(t, Um[:, 0], "-", color=cols[label], lw=2.0,
                   label=f"{label}, main engine")
        ax[0].plot(t, Um[:, 1], "--", color=cols[label], lw=1.5,
                   label=f"{label}, wheel torque")
    ax[0].axhline(0.0, color="k", lw=0.8)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("thrust [N]")
    ax[0].set_title("Quadratic trickles, propellant burns and coasts")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    for label, d in runs.items():
        ax[1].plot(d["Z"][:, 0], d["Z"][:, 1], "-", color=cols[label], lw=2.0,
                   label=label)
    ax[1].plot([target[0]], [target[1]], "*", color="steelblue", ms=14,
               label="target")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("y [m]")
    for label, d in runs.items():
        Zd = d["Z"][::4]
        ax[1].quiver(Zd[:, 0], Zd[:, 1], np.cos(Zd[:, 2]), np.sin(Zd[:, 2]),
                     color=cols[label], scale=22, width=0.004, alpha=0.8)
    ax[1].set_title("Path flown, arrows show where it points")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].set_aspect("equal", adjustable="datalim")

    for label, d in runs.items():
        Um = d["U"].reshape(N, nu)
        burn = np.cumsum(np.sum(A_PROP * np.abs(Um) + B_PROP * Um ** 2,
                                axis=1))
        ax[2].plot(t, burn, "-", color=cols[label], lw=2.0,
                   label=f"{label}: {burn[-1]:.3g}")
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("propellant burned, cumulative")
    ax[2].set_title("Where the propellant goes")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/tests/nq_cost.png", dpi=140, bbox_inches="tight")
    print("\nsaved figures/tests/nq_cost.png")