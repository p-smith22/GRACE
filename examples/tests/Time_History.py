import os, time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grace

# Cart-pole dynamics (pole hangs at theta = 0, upright at theta = pi):
def cartpole(x, u, l=0.5):
    g = 9.81
    mc = 1.0
    mp = 0.1
    s = ca.sin(x[1])
    c = ca.cos(x[1])
    den = mc + mp * s * s
    ddx = (u[0] + mp * s * (l * x[3] ** 2 + g * c)) / den
    ddth = (-u[0] * c - mp * l * x[3] ** 2 * c * s - (mc + mp) * g * s) / (l * den)
    return ca.vertcat(x[2], x[3], ddx, ddth)

job_name = "cartpole_swingup"
os.makedirs(f"figures/{job_name}", exist_ok=True)

N, dt = 30, 0.05
system = grace.build_cached(cartpole, nx=4, nu=1, N=N, z0=[0,0,0,0], dt=dt, job=job_name)
engine = grace.GRACE(system)
n = N * system.nu
target = np.array([0.0, np.pi, 0.0, 0.0])

S = np.tril(np.ones((n, n)))
eps = 0.02
phi_p  = lambda a: a + eps*a**3
phi_pp = lambda a: 1.0 + 3*eps*a**2

def f(U):
    a = S @ np.asarray(U).ravel()
    return float(np.sum(0.5*a**2 + 0.25*eps*a**4))

def grad(U):
    a = S @ np.asarray(U).ravel()
    return S.T @ phi_p(a)

def hess(U):
    a = S @ np.asarray(U).ravel()
    return S.T @ (phi_pp(a)[:, None] * S)

def dinv(v):
    v = np.asarray(v).ravel(); U = np.zeros(n)
    for _ in range(80):
        r = grad(U) - v
        if np.linalg.norm(r) < 1e-10: break
        U -= np.linalg.solve(hess(U) + 1e-9*np.eye(n), r)
    return U

def dprime(v):
    return np.linalg.inv(hess(dinv(v)) + 1e-9*np.eye(n))

cost = (f, dinv, dprime, grad)

start = time.time()
Ug = np.asarray(engine.shooting.lambda_shoot(target, cost=cost)).ravel()
print(f"SWING-UP ({time.time()-start:.2f}s)  {engine.utils.diagnostics(Ug, target)}")

# IPOPT baseline: hand SX RK4 tape, verified against system.rollout:
zs = ca.SX.sym("z", 4); us = ca.SX.sym("u", 1)
k1 = cartpole(zs, us); k2 = cartpole(zs+0.5*dt*k1, us)
k3 = cartpole(zs+0.5*dt*k2, us); k4 = cartpole(zs+dt*k3, us)
step = ca.Function("step", [zs, us], [zs + (dt/6.0)*(k1+2*k2+2*k3+k4)])
Uv = ca.MX.sym("U", n)
zc = ca.MX(np.zeros(4))
for k in range(N):
    zc = step(zc, Uv[k])
chk = np.asarray(step.map(N)(np.zeros((4,N)), np.zeros((1,N)))[:, -1]).ravel()
assert np.allclose(chk, np.asarray(system.rollout(np.zeros(n))[-1]).ravel())
a = ca.mtimes(ca.DM(S), Uv)
nlp = {"x": Uv, "f": ca.sum1(0.5*a**2 + 0.25*eps*a**4), "g": zc - ca.DM(target)}
sol = ca.nlpsol("s", "ipopt", nlp, {"ipopt.tol":1e-10, "ipopt.print_level":0, "print_time":0,
                                     "ipopt.max_iter":3000})
r = sol(x0=Ug, lbg=0, ubg=0)
Ui = np.asarray(r["x"]).ravel()

Jg, Ji = f(Ug), f(Ui)
gap = abs(Jg-Ji)/abs(Ji)
ctrl_err = np.linalg.norm(Ug-Ui)/np.linalg.norm(Ui)
print("cost   GRACE %.9f  IPOPT %.9f  gap %.3e" % (Jg, Ji, gap))
print("control ||dU||/||U|| %.3e" % ctrl_err)
eg = np.linalg.norm(np.asarray(system.rollout(Ug)[-1]).ravel() - target)
ei = np.linalg.norm(np.asarray(system.rollout(Ui)[-1]).ravel() - target)
print("endpoint err  GRACE %.2e  IPOPT %.2e" % (eg, ei))

Zg = np.asarray(system.rollout(Ug)).reshape(N+1, 4)
Zi = np.asarray(system.rollout(Ui)).reshape(N+1, 4)
t = np.arange(N+1) * dt
tu = np.arange(N) * dt

fig, axs = plt.subplots(3, 1, figsize=(6.2, 7.2), sharex=True)
axs[0].plot(t, Zg[:,0], 'o-', ms=3, color="#1f77b4", label="GRACE")
axs[0].plot(t, Zi[:,0], 'x--', ms=5, color="#d62728", label="IPOPT")
axs[0].set_ylabel("cart position  x")
axs[0].legend(loc="best", fontsize=8)

axs[1].plot(t, Zg[:,1], 'o-', ms=3, color="#1f77b4")
axs[1].plot(t, Zi[:,1], 'x--', ms=5, color="#d62728")
axs[1].axhline(np.pi, color="gray", ls=":", lw=1, label=r"target $\theta=\pi$")
axs[1].set_ylabel(r"pole angle  $\theta$")
axs[1].legend(loc="best", fontsize=8)

axs[2].step(tu, Ug, where="post", color="#1f77b4", label="GRACE")
axs[2].step(tu, Ui, where="post", color="#d62728", ls="--", label="IPOPT")
axs[2].set_ylabel("control  u")
axs[2].set_xlabel("time [s]")
axs[2].legend(loc="best", fontsize=8)

fig.suptitle("Swing-up, dense whole-history cost  ($N=%d$)\n"
             r"cost gap %.1e   $\|\Delta U\|/\|U\|$ %.1e" % (N, gap, ctrl_err), fontsize=10)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig("figures/tests/Time_History.png", dpi=160)
print("saved figure")