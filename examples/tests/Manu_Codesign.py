# Import packages:
import time
import numpy as np
import casadi as ca
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace

# === PLANT ===
# Both axes are deliberately short of authority. A strong aileron would cover
# roll at any cant and a strong rudder would cover yaw at any cant, and in
# either case the winglet's orientation would stop mattering -- the design
# conflict only exists when the winglet carries a real share of both:
IXX, IZZ = 4.0, 6.5
L_A, L_R = 0.25, 0.05
N_A, N_R = -0.08, 0.55
Y_B, Y_R = -0.20, 0.20
D_P, D_R = 1.6, 1.1
V_TAS, G_ACC = 45.0, 9.81
L_REF = 10.0

# Winglet authority at full deflection, before the cant angle splits it:
L_W0, N_W0, Y_W0 = 0.60, 0.80, -0.30

# Span and winglet length, for the structural objective below:
B_SPAN, L_WING = 10.0, 1.2

# Cant enters as a rotation of the winglet's force and moment contributions.
# One parameter, three derivatives, and the split is trigonometric rather than
# fitted -- a surface at angle Lambda from horizontal contributes cos(Lambda)
# of its authority to roll and sin(Lambda) to yaw and side force:
def dynamics(z, u, p):
    phi, pr, psi, r, beta = z[0], z[1], z[2], z[3], z[4]
    da, dw, dr = u[0], u[1], u[2]
    lam = p[0]
    cl, sl = ca.cos(lam), ca.sin(lam)
    return ca.vertcat(
        pr,
        (L_A * da + L_W0 * cl * dw + L_R * dr
         - D_P * pr * ca.sqrt(pr ** 2 + 1e-6)) / IXX,
        r,
        (N_A * da + N_W0 * sl * dw + N_R * dr
         - D_R * r * ca.sqrt(r ** 2 + 1e-6)) / IZZ,
        (G_ACC / V_TAS) * phi - r + Y_B * beta + Y_W0 * sl * dw + Y_R * dr,
        V_TAS * (psi + beta) / L_REF)

nx, nu = 6, 3
T_MAN = 8.0
N, dt = 40, T_MAN / 40
z0 = np.zeros(nx)
ZNAMES = ("phi", "p", "psi", "r", "beta", "y/L")

# === MANEUVERS ===
# The pair has to separate cleanly on the roll-yaw axis or there is no design
# conflict to show. A bank and hold needs roll and nothing else; a flat turn
# needs heading with the wings level, which an aileron cannot supply.
#
# Both constrain three endpoint components and both reach their target through
# one integration. Constraining the flat turn's sideslip and lateral position
# as well put its target at the end of a double integration through bank, and
# the inner problem became multimodal: a pinned design comparison against
# IPOPT showed the two solvers landing in different basins, GRACE cheaper at
# some designs and dearer at others with both feasible to machine precision.
# Two feasible controls cannot both be minimal, so the whole flat turn column
# was unreliable. Trimming to three components makes the pair symmetric and
# neither problem harder than the other:
PHI_DEG = 20.0
PSI_DEG = 12.0
_MAN = {
    "bank and hold": dict(
        target=np.array([np.deg2rad(PHI_DEG), 0.0, 0.0, 0.0, 0.0, 0.0]),
        tidx=[0, 1, 4]),
    "flat turn": dict(
        target=np.array([0.0, 0.0, np.deg2rad(PSI_DEG), 0.0, 0.0, 0.0]),
        tidx=[2, 3, 0])}

# Order the shared-design problem builds its control blocks in. Swapping this
# is a one line test of whether a per block fault follows the maneuver or its
# position in the build loop:
MAN_ORDER = ["bank and hold", "flat turn"]
MANEUVERS = {k: _MAN[k] for k in MAN_ORDER}

# === DESIGN OBJECTIVE ===
# Root bending moment. A winglet laid flat extends the effective span and
# loads the wing root; canting it up recovers most of the aerodynamic benefit
# with far less bending, which is why winglets exist rather than plain span
# extensions. Normalized to vanish at full cant:
def bending(lam):
    b_eff = B_SPAN + 2.0 * L_WING * np.cos(float(lam))
    return float((b_eff / B_SPAN) ** 2 - 1.0)

# Cant angle range. Below about ten degrees the surface is a span extension
# and above eighty it is a fin, so the bounds are where the model stops being
# a winglet:
LAM_LO, LAM_HI = np.deg2rad(10.0), np.deg2rad(80.0)
LAM_SPAN = LAM_HI - LAM_LO
LAM_0 = 0.5 * (LAM_LO + LAM_HI)

# Weights traced. The endpoints need no interpretation: w = 0 minimizes
# control effort alone and w = 1 minimizes bending alone, so the extremes are
# unambiguous and the interior is the trade between them:
W_VALUES = np.linspace(0.0, 1.0, 21)

# Background curve, for the figure only. Nothing about the located designs
# depends on this spacing, since they come from a root find:
N_CURVE = 25

# Common IPOPT settings:
_IP_OPTS = dict(ipopt=dict(print_level=0, sb="yes", tol=1e-12,
                           dual_inf_tol=1e-10, constr_viol_tol=1e-12,
                           compl_inf_tol=1e-10, acceptable_tol=1e-12,
                           acceptable_iter=999, mu_strategy="adaptive",
                           max_iter=3000),
                print_time=0)

# One RK4 step carrying the design parameter:
def _step(tag):
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    pp = ca.MX.sym("p", 1)
    k1 = dynamics(z, u, pp)
    k2 = dynamics(z + 0.5 * dt * k1, u, pp)
    k3 = dynamics(z + 0.5 * dt * k2, u, pp)
    k4 = dynamics(z + dt * k3, u, pp)
    return ca.Function(f"st_{tag}", [z, u, pp],
                       [z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

# Rollout endpoint at a design and control, outside any solver, so a returned
# solution can be checked against the constraint it was supposed to satisfy:
def endpoint_of(U, lam_val):
    step = _step("chk")
    Zc = step.mapaccum("roll_chk", N)(
        ca.DM(z0), ca.reshape(ca.DM(np.asarray(U, float)), nu, N),
        ca.repmat(ca.DM([float(lam_val)]), 1, N))
    return np.array(Zc)[:, -1]

# === INDEPENDENT REFERENCE ===
# IPOPT on the full design problem: the design enters as a decision variable
# alongside the controls, with the endpoint as a constraint and the design
# objective in the cost. Nothing here uses GRACE, so agreement is a check on
# the design gradient rather than on the search. The design is carried as a
# unit variable so it sits on the same footing as the controls.
#
# lam_fix pins the design by collapsing its bounds. That turns the same
# function into a pure inner solve, whose cost is directly comparable to a
# minimum effort shoot at that design -- without it the design is free and the
# solver returns its own optimum whatever value it was started from, which
# compares two designs rather than two solvers:
def ipopt_design(target, tidx, w, nrm, lam_guess=None, lam_fix=None):
    step = _step("d")
    U = ca.MX.sym("U", N * nu)
    s = ca.MX.sym("s", 1)
    lam = LAM_LO + LAM_SPAN * s
    Zc = step.mapaccum("roll_d", N)(ca.DM(z0), ca.reshape(U, nu, N),
                                    ca.repmat(lam, 1, N))
    g_end = Zc[tidx, -1] - ca.DM(np.asarray(target, float)[tidx])

    # Both objectives range-normalized from the ideal point, exactly as the
    # front solve normalizes them, so the same weight means the same trade:
    Chat = (ca.dot(U, U) - nrm["C_id"]) / nrm["C_rng"]
    b_eff = B_SPAN + 2.0 * L_WING * ca.cos(lam)
    Dhat = (((b_eff / B_SPAN) ** 2 - 1.0) - nrm["D_id"]) / nrm["D_rng"]
    f = (1.0 - w) * Chat + w * Dhat

    X = ca.vertcat(U, s)
    nlp = ca.nlpsol("dz", "ipopt", dict(x=X, f=f, g=g_end), _IP_OPTS)
    ng = g_end.shape[0]

    # Starting design, and the bounds it is allowed to move within:
    s0 = ((LAM_0 if lam_guess is None else float(lam_guess))
          - LAM_LO) / LAM_SPAN
    if lam_fix is None:
        s_lo, s_hi = 0.0, 1.0
    else:
        s0 = (float(lam_fix) - LAM_LO) / LAM_SPAN
        s_lo, s_hi = s0, s0

    x0 = np.concatenate([np.zeros(N * nu), [s0]])
    lbx = np.concatenate([-np.inf * np.ones(N * nu), [s_lo]])
    ubx = np.concatenate([np.inf * np.ones(N * nu), [s_hi]])
    t0 = time.perf_counter()
    sol = nlp(x0=x0, lbx=lbx, ubx=ubx, lbg=np.zeros(ng), ubg=np.zeros(ng))
    t = time.perf_counter() - t0
    xv = np.array(sol["x"]).flatten()
    return dict(lam=LAM_LO + LAM_SPAN * float(xv[-1]), U=xv[:-1],
                cost=float(xv[:-1] @ xv[:-1]), f=float(sol["f"]), t=t,
                g_res=float(np.linalg.norm(np.array(sol["g"]).flatten())),
                ok=bool(nlp.stats()["success"]), its=nlp.stats()["iter_count"])

# Two maneuvers sharing one design. Each carries its own control block and its
# own endpoint constraint, and the objective averages their span-scaled
# efforts against the same design term. Per block costs, controls and
# constraint residuals are all returned, so a block that disagrees with a
# minimum effort solve at the same design can be traced to a constraint it
# missed or to a genuinely different control:
def ipopt_balance(w, span, D_lo, D_hi, lam_guess=None):
    s = ca.MX.sym("s", 1)
    lam = LAM_LO + LAM_SPAN * s
    Us, gs, obj, sizes = [], [], 0.0, []

    for j, name in enumerate(MAN_ORDER):
        mv = MANEUVERS[name]
        step = _step(f"b{j}")
        Uk = ca.MX.sym(f"Ub_{j}", N * nu)
        Zc = step.mapaccum(f"rollb_{j}", N)(ca.DM(z0), ca.reshape(Uk, nu, N),
                                            ca.repmat(lam, 1, N))
        gk = Zc[mv["tidx"], -1] \
            - ca.DM(np.asarray(mv["target"], float)[mv["tidx"]])
        gs.append(gk)
        sizes.append(gk.shape[0])

        # Same span scaling the balance uses, so both maneuvers enter with
        # equal say and the two objectives are identical term by term:
        lo_j, hi_j = span[name]
        obj = obj + (1.0 - w) * ((ca.dot(Uk, Uk) - lo_j)
                                 / max(hi_j - lo_j, 1e-30)) / len(MAN_ORDER)
        Us.append(Uk)

    # Design term added once, outside the loop, matching obj_balance:
    b_eff = B_SPAN + 2.0 * L_WING * ca.cos(lam)
    obj = obj + w * (((b_eff / B_SPAN) ** 2 - 1.0 - D_lo)
                     / max(D_hi - D_lo, 1e-30))

    X = ca.vertcat(*Us, s)
    G = ca.vertcat(*gs)
    nlp = ca.nlpsol("bz", "ipopt", dict(x=X, f=obj, g=G), _IP_OPTS)
    ng = G.shape[0]
    nU = len(Us) * N * nu
    l0 = LAM_0 if lam_guess is None else float(lam_guess)
    x0 = np.concatenate([np.zeros(nU), [(l0 - LAM_LO) / LAM_SPAN]])
    lbx = np.concatenate([-np.inf * np.ones(nU), [0.0]])
    ubx = np.concatenate([np.inf * np.ones(nU), [1.0]])
    t0 = time.perf_counter()
    sol = nlp(x0=x0, lbx=lbx, ubx=ubx, lbg=np.zeros(ng), ubg=np.zeros(ng))
    t = time.perf_counter() - t0
    xv = np.array(sol["x"]).flatten()
    gv = np.array(sol["g"]).flatten()

    # Split the residual back onto the block that produced it:
    blocks, off = {}, 0
    for j, name in enumerate(MAN_ORDER):
        Uj = xv[j * N * nu:(j + 1) * N * nu]
        blocks[name] = dict(U=Uj, cost=float(Uj @ Uj), pos=j,
                            g_res=float(np.linalg.norm(gv[off:off+sizes[j]])))
        off += sizes[j]

    return dict(lam=LAM_LO + LAM_SPAN * float(xv[-1]), f=float(sol["f"]),
                blocks=blocks, t=t,
                g_res=float(np.linalg.norm(gv)),
                ok=bool(nlp.stats()["success"]), its=nlp.stats()["iter_count"])

# === RUN ===
if __name__ == "__main__":
    lams_c = np.linspace(LAM_LO, LAM_HI, N_CURVE)
    ld = np.rad2deg(lams_c)
    names = list(MAN_ORDER)
    cases = names + ["balance"]

    # Identity normalization, for solves where the raw control cost is wanted
    # rather than a scalarization:
    RAW = dict(C_id=0.0, C_rng=1.0, D_id=0.0, D_rng=1.0)

    print(f"winglet cant codesign, {np.rad2deg(LAM_LO):.0f} to "
          f"{np.rad2deg(LAM_HI):.0f} deg, {len(W_VALUES)} weights")
    print(f"build order {MAN_ORDER}")
    print(f"w = 0 minimizes control effort alone, "
          f"w = 1 minimizes bending alone\n")

    binder = {n: grace.Codesign(dynamics, nx=nx, nu=nu, N=N, z0=z0, dt=dt)
              for n in names}

    # Control cost at one design. Memoized on the design value, since the
    # balance search revisits points and every miss is a full solve:
    _ccache = {}

    def cost_at(name, lam_val):
        key = (name, round(float(lam_val), 12))
        if key not in _ccache:
            fr = binder[name].scan(MANEUVERS[name]["target"], "cant", bending,
                                   [float(lam_val)],
                                   target_idx=MANEUVERS[name]["tidx"],
                                   plot=False, filter_dominated=False)
            _ccache[key] = float(fr[0]["cost"])
        return _ccache[key]

    # === INNER SOLVE CHECK ===
    # Minimum effort at a pinned design, GRACE against IPOPT on the identical
    # problem. Everything downstream reads the inner solve as the minimum
    # effort cost at a design, so if the two disagree here nothing built on
    # top of it means anything. A gap of either sign with both solutions
    # feasible means the inner problem has more than one local minimum, since
    # two feasible controls cannot both be minimal:
    print("inner solve at a pinned design, GRACE against IPOPT")
    print(f"{'maneuver':<16}{'cant':>8}{'GRACE':>12}{'IPOPT':>12}"
          f"{'gap':>10}{'IP g res':>11}")
    for lam_t in np.deg2rad([31.91, 45.0, 60.0, 80.0]):
        for name in names:
            mv = MANEUVERS[name]
            r = ipopt_design(mv["target"], mv["tidx"], 0.0, RAW,
                             lam_fix=lam_t)
            c_g = cost_at(name, lam_t)
            print(f"{name:<16}{np.rad2deg(lam_t):>8.2f}{c_g:>12.6f}"
                  f"{r['cost']:>12.6f}"
                  f"{(c_g / max(r['cost'], 1e-30) - 1.0) * 100:>9.2f}%"
                  f"{r['g_res']:>11.2e}")
    print()

    # === GRADIENT CHECK ===
    # Central differences of the control cost. The design solve roots the same
    # derivative, so where this crosses zero is where the control-only design
    # has to land:
    print("control cost derivative by central differences")
    print(f"{'maneuver':<16}" + "".join(f"{f'{d:.0f} deg':>12}"
                                        for d in (25.0, 45.0, 65.0)))
    h_fd = 1e-4
    for name in names:
        row = [(cost_at(name, l + h_fd) - cost_at(name, l - h_fd))
               / (2.0 * h_fd) for l in np.deg2rad([25.0, 45.0, 65.0])]
        print(f"{name:<16}" + "".join(f"{v:>12.4g}" for v in row))

    # === DESIGN SOLVE ===
    res, t_grace = {}, {}
    for name in names:
        mv = MANEUVERS[name]
        t0 = time.perf_counter()
        _, _, pareto, sweep = binder[name].optimize(
            mv["target"], "cant", bending, LAM_0, (LAM_LO, LAM_HI),
            weights=W_VALUES, target_idx=mv["tidx"], norm="l1",
            plot=False, filter_dominated=False,
            job=f"cant_{name.replace(' ', '_')}")
        t_grace[name] = time.perf_counter() - t0
        res[name] = dict(front=pareto, sweep=sweep)
        print(f"\n{name:<16} {len(pareto)} front points, control-only "
              f"{np.rad2deg(pareto[0]['param']):.2f} deg, bending-only "
              f"{np.rad2deg(pareto[-1]['param']):.2f} deg, "
              f"{t_grace[name]:.2f} s")

    # === BALANCE ===
    # Each maneuver's cost is scaled to its own span over the design box
    # before averaging, so the two enter with equal say. The front machinery
    # covers a single maneuver, so this is a scalar minimization evaluated by
    # solving both maneuvers at the candidate design:
    span = {}
    for name in names:
        Cs = np.array([cost_at(name, l) for l in lams_c])
        span[name] = (float(Cs.min()), float(Cs.max()))
        print(f"[balance] {name}: cost spans {span[name][0]:.4f} to "
              f"{span[name][1]:.4f}")

    D_lo = min(bending(l) for l in lams_c)
    D_hi = max(bending(l) for l in lams_c)

    def obj_balance(lam_val, w):
        Chat = sum((cost_at(n, lam_val) - span[n][0])
                   / max(span[n][1] - span[n][0], 1e-30)
                   for n in names) / len(names)
        Dhat = (bending(lam_val) - D_lo) / max(D_hi - D_lo, 1e-30)
        return (1.0 - w) * Chat + w * Dhat

    bal = []
    for w in W_VALUES:
        r = minimize_scalar(lambda q: obj_balance(q, float(w)),
                            bounds=(LAM_LO, LAM_HI), method="bounded",
                            options=dict(xatol=1e-8))
        bal.append(dict(weight=float(w), param=float(r.x),
                        objective_total=float(r.fun)))
    print(f"\n{'balance':<16} control-only "
          f"{np.rad2deg(bal[0]['param']):.2f} deg, bending-only "
          f"{np.rad2deg(bal[-1]['param']):.2f} deg")

    # === DESIGN AGAINST WEIGHT ===
    print(f"\ndesign [deg] against weight")
    print(f"{'w':>6}" + "".join(f"{c:>18}" for c in cases))
    for i in range(0, len(W_VALUES), max(1, len(W_VALUES) // 8)):
        print(f"{res[names[0]]['front'][i]['weight']:>6.2f}"
              + "".join(f"{np.rad2deg(res[n]['front'][i]['param']):>18.2f}"
                        for n in names)
              + f"{np.rad2deg(bal[i]['param']):>18.2f}")

    sep = np.array([abs(res[names[0]]["front"][i]["param"]
                        - res[names[1]]["front"][i]["param"])
                    for i in range(len(W_VALUES))])
    print(f"\nseparation {np.rad2deg(sep.max()):.2f} deg at w = "
          f"{W_VALUES[int(np.argmax(sep))]:.2f}, "
          f"{np.rad2deg(sep[-1]):.2f} deg at w = 1")

    # === CHECK ===
    print(f"\nindependent check, IPOPT on the full design problem [deg]")
    print(f"{'case':<16}{'w':>6}{'GRACE':>10}{'IPOPT':>10}{'diff':>9}"
          f"{'g res':>11}{'its':>6}{'ok':>7}")
    ip, ip_bal = {}, {}
    for name in names:
        mv = MANEUVERS[name]
        ip[name] = {}
        for w_t in (0.0, 0.5, 1.0):
            k = int(np.argmin(np.abs(W_VALUES - w_t)))
            fk = res[name]["front"][k]
            r = ipopt_design(mv["target"], mv["tidx"], fk["weight"],
                             fk["norm"])
            ip[name][w_t] = r
            print(f"{name:<16}{fk['weight']:>6.2f}"
                  f"{np.rad2deg(fk['param']):>10.2f}"
                  f"{np.rad2deg(r['lam']):>10.2f}"
                  f"{np.rad2deg(r['lam'] - fk['param']):>9.2f}"
                  f"{r['g_res']:>11.2e}{r['its']:>6}{str(r['ok']):>7}")
    for w_t in (0.0, 0.5, 1.0):
        k = int(np.argmin(np.abs(W_VALUES - w_t)))
        r = ipopt_balance(float(W_VALUES[k]), span, D_lo, D_hi,
                          lam_guess=bal[k]["param"])
        ip_bal[w_t] = r
        print(f"{'balance':<16}{W_VALUES[k]:>6.2f}"
              f"{np.rad2deg(bal[k]['param']):>10.2f}"
              f"{np.rad2deg(r['lam']):>10.2f}"
              f"{np.rad2deg(r['lam'] - bal[k]['param']):>9.2f}"
              f"{r['g_res']:>11.2e}{r['its']:>6}{str(r['ok']):>7}")

    # === WHERE A DISAGREEMENT COMES FROM ===
    # Whether the two objectives are the same function, from IPOPT's own value
    # against this objective at IPOPT's own design. Whether each control block
    # is at its own minimum for the design returned. Whether each block met
    # its constraints, per block rather than summed. And what endpoint each
    # block actually reached:
    print(f"\nwhere a balance disagreement comes from")
    for w_t in (0.0, 0.5):
        k = int(np.argmin(np.abs(W_VALUES - w_t)))
        r = ip_bal[w_t]
        o_g = obj_balance(bal[k]["param"], float(W_VALUES[k]))
        o_i = obj_balance(r["lam"], float(W_VALUES[k]))
        print(f"\n  w = {W_VALUES[k]:.2f}, GRACE "
              f"{np.rad2deg(bal[k]['param']):.2f} deg -> {o_g:.8f}, "
              f"IPOPT {np.rad2deg(r['lam']):.2f} deg -> {o_i:.8f}")
        print(f"    IPOPT own f {r['f']:.8f}, objective gap "
              f"{abs(o_i - r['f']):.2e}")
        for name in names:
            b = r["blocks"][name]
            c_g = cost_at(name, r["lam"])
            print(f"    {name:<16} block {b['pos']}, cost {b['cost']:.6f} "
                  f"vs min effort {c_g:.6f}, excess "
                  f"{(b['cost'] / c_g - 1.0) * 100:+.3f}%, g res "
                  f"{b['g_res']:.2e}")
            e_ip = endpoint_of(b["U"], r["lam"])
            tgt = np.asarray(MANEUVERS[name]["target"], float)
            miss = [f"{ZNAMES[i]} {e_ip[i] - tgt[i]:+.2e}"
                    for i in MANEUVERS[name]["tidx"]]
            print(f"      endpoint miss: " + ", ".join(miss))

    # === PENALTY FOR THE WRONG DESIGN ===
    print(f"\ncontrol cost at each control-only design, relative to own best")
    at = {n: res[n]["front"][0]["param"] for n in names}
    at["balance"] = bal[0]["param"]
    print(f"{'maneuver':<16}" + "".join(f"{'sized for ' + c:>22}"
                                        for c in cases))
    pen = {}
    for name in names:
        own = res[name]["front"][0]["cost"]
        pen[name] = [cost_at(name, at[c]) / own for c in cases]
        print(f"{name:<16}" + "".join(f"{v:>22.3f}" for v in pen[name]))

    # === PLOT ===
    cols = {"bank and hold": "crimson", "flat turn": "steelblue",
            "balance": "0.35"}
    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.6))

    for name in names:
        ax[0].plot([f["weight"] for f in res[name]["front"]],
                   [np.rad2deg(f["param"]) for f in res[name]["front"]],
                   "-o", color=cols[name], lw=2.2, ms=3.5, label=name)
        for w_t, r in ip[name].items():
            ax[0].plot(w_t, np.rad2deg(r["lam"]), "x", color=cols[name],
                       ms=10, mew=2.2)
    ax[0].plot([b["weight"] for b in bal],
               [np.rad2deg(b["param"]) for b in bal], "--s",
               color=cols["balance"], lw=2.0, ms=3.5, label="balance")
    for w_t, r in ip_bal.items():
        ax[0].plot(w_t, np.rad2deg(r["lam"]), "x", color=cols["balance"],
                   ms=10, mew=2.2)
    ax[0].fill_between([f["weight"] for f in res[names[0]]["front"]],
                       [np.rad2deg(f["param"])
                        for f in res[names[0]]["front"]],
                       [np.rad2deg(f["param"])
                        for f in res[names[1]]["front"]],
                       color="0.6", alpha=0.15)
    ax[0].plot([], [], "kx", ms=9, mew=2.0, label="IPOPT")
    ax[0].set_xlabel("Weight, Effort -> Design")
    ax[0].set_ylabel("Dihedral [deg]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    curves = {}
    for name in names:
        C = np.array([cost_at(name, l) for l in lams_c])
        lo_j, hi_j = span[name]
        curves[name] = (C - lo_j) / max(hi_j - lo_j, 1e-30)
    curves["balance"] = sum(curves[n] for n in names) / len(names)

    for lab in cases:
        ax[1].plot(ld, curves[lab], "--" if lab == "balance" else "-",
                   color=cols[lab], lw=2.2, label=lab)
        p_w0 = (bal[0]["param"] if lab == "balance"
                else res[lab]["front"][0]["param"])
        ax[1].plot(np.rad2deg(p_w0),
                   np.interp(np.rad2deg(p_w0), ld, curves[lab]), "o",
                   color=cols[lab], ms=9)
        l_ip = (ip_bal[0.0]["lam"] if lab == "balance"
                else ip[lab][0.0]["lam"])
        ax[1].plot(np.rad2deg(l_ip),
                   np.interp(np.rad2deg(l_ip), ld, curves[lab]), "x",
                   color=cols[lab], ms=11, mew=2.4)
    ax[1].plot([], [], "kx", ms=9, mew=2.0, label="IPOPT")
    ax[1].set_xlabel("Dihedral [deg]")
    ax[1].set_ylabel("Control Cost")
    ax[1].set_title("Weight: $w = 0$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    xb = np.arange(len(names))
    wbar = 0.26
    for k, lab in enumerate(cases):
        ax[2].bar(xb + (k - 1) * wbar, [pen[n][k] for n in names], wbar,
                  color=cols[lab], label=f"sized for {lab}")
    ax[2].axhline(1.0, color="k", lw=1.0)
    ax[2].set_xticks(xb)
    ax[2].set_xticklabels(names)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("Relative Control Cost")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    fig.savefig("figures/tests/maneuver_codesign.png", dpi=140,
                bbox_inches="tight")
    print("\nsaved figures/tests/maneuver_codesign.png")