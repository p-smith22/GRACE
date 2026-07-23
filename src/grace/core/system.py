# ============================================================================
# system.py -- the compiled system that every GRACE module runs on:
# ============================================================================
# A System holds the CasADi-compiled functions built from a user's dynamics:
# the endpoint map and its analytic Jacobian, the full rollout, the one-step
# Jacobians for tracking, and the position Jacobian for obstacle avoidance.
# Because compilation is the slow part, a System can save its compiled graph
# to the data directory under a job name and reload it on later runs.
# ============================================================================

import os
import numpy as np
import casadi as ca


# Default directory for cached rollout graphs:
DATA_DIR = "data"


class System:

    # Hold the compiled functions and problem dimensions for one dynamics model:
    def __init__(self, F_end, F_roll, step, step_jac, nx, nu, N, z0, dt,
                 target_idx=None, pos_jac=None, job=None, pos_idx=None):

        # Store the compiled CasADi functions:
        self.F_end = F_end                      # endpoint map U -> [g, dg/dU]
        self.F_roll = F_roll                    # full rollout U -> Z (N+1, nx)
        self.step = step                        # one-step dynamics z, u -> z_next
        self.step_jac = step_jac                # one-step Jacobians z, u -> [A, B]
        self.step_jac_all = None                # batched Jacobians over the whole tape
        self.pos_jac = pos_jac                  # position Jacobian U -> dPos/dU

        # Store the problem dimensions:
        self.nx = nx                            # state dimension
        self.nu = nu                            # control dimension
        self.N = N                              # number of horizon steps
        self.z0 = np.asarray(z0, float)         # initial state
        self.dt = dt                            # time step

        # Store which endpoint components are constrained (default all):
        self.tidx = list(range(nx)) if target_idx is None else list(target_idx)
        self.m = len(self.tidx)                 # number of endpoint constraints

        # Store the job name used for graph caching:
        self.job = job

        # Store the position indices used for obstacle avoidance (for caching):
        self.pos_idx = list(pos_idx) if pos_idx is not None else None

        # Last-value caches for the rollout and endpoint+Jacobian, keyed on the
        # control tape so repeated evaluations on the same U are free:
        self._roll_cache = None
        self._endjac_cache = None

    # Roll the control tape out to the full state trajectory:
    def rollout(self, U):

        # Return the cached rollout when the control tape is unchanged.  Solvers call
        # rollout, endpoint, and endpoint_jac repeatedly on the SAME U within one
        # iteration (line searches, feasibility checks), so a last-value cache keyed
        # on the control tape avoids recompiled-graph re-evaluations:
        Uf = np.asarray(U, float).flatten()
        z0 = np.asarray(self.z0, float)
        # The cache key includes the anchor state: the same control tape gives a
        # different trajectory when the vehicle starts somewhere else, so a cache
        # keyed only on U would return a stale plan after re-anchoring.
        hit = self._roll_cache
        if (hit is not None and hit[0].shape == Uf.shape
                and np.array_equal(hit[0], Uf) and np.array_equal(hit[2], z0)):
            return hit[1]

        # Evaluate the compiled rollout, cache it, and return a plain array:
        Z = np.array(self.F_roll(Uf, z0))
        self._roll_cache = (Uf.copy(), Z, z0.copy())
        return Z

    # Evaluate the endpoint and its analytic Jacobian at a control tape:
    def endpoint_jac(self, U):

        # Return the cached endpoint+Jacobian when the control tape is unchanged:
        Uf = np.asarray(U, float).flatten()
        z0 = np.asarray(self.z0, float)
        # As with the rollout cache, the anchor state is part of the key:
        hit = self._endjac_cache
        if (hit is not None and hit[0].shape == Uf.shape
                and np.array_equal(hit[0], Uf) and np.array_equal(hit[3], z0)):
            return hit[1], hit[2]

        # Evaluate the compiled endpoint map, cache, and return plain arrays:
        g, J = self.F_end(Uf, z0)
        g = np.array(g).flatten()
        J = np.array(J)
        self._endjac_cache = (Uf.copy(), g, J, z0.copy())
        return g, J

    # Evaluate the endpoint only:
    def endpoint(self, U):

        # Reuse the endpoint+Jacobian cache (dropping the Jacobian) so an endpoint()
        # call right after endpoint_jac() on the same U is free:
        g, _ = self.endpoint_jac(U)
        return g

    # Reduce a full state target to the constrained components:
    def target(self, z_target):

        # Take only the constrained indices if a full state was given:
        zt = np.asarray(z_target, float)
        return zt[self.tidx] if len(zt) == self.nx else zt

    # Evaluate the numeric one-step dynamics (used by the closed-loop simulate):
    def step_np(self, z, u):

        # Evaluate the compiled step and return a plain array:
        return np.array(self.step(z, u)).flatten()


# Build a compiled System from a user's dynamics function:
def build(dynamics, nx, nu, N, z0, dt, target_idx=None, pos_idx=None,
          substeps=1, jit=True, job=None, data_dir=DATA_DIR, rebuild=False):

    # The dynamics function maps symbolic (x, u) to xdot as CasADi expressions.
    # Everything GRACE needs is derived from it by RK4 integration and AD.

    # Symbolic state and control for the one-step map:
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)

    # Integrate one control period with RK4 over the requested substeps:
    h = dt / substeps
    zc = z
    for _ in range(substeps):
        k1 = dynamics(zc, u)
        k2 = dynamics(zc + 0.5 * h * k1, u)
        k3 = dynamics(zc + 0.5 * h * k2, u)
        k4 = dynamics(zc + h * k3, u)
        zc = zc + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Compile the one-step map and its Jacobians:
    step = ca.Function("step", [z, u], [zc])
    A = ca.Function("A", [z, u], [ca.jacobian(zc, z)])
    B = ca.Function("B", [z, u], [ca.jacobian(zc, u)])

    # Wrap the one-step Jacobians so callers get plain arrays:
    def step_jac(zv, uv):
        return np.array(A(zv, uv)), np.array(B(zv, uv))

    # Batched version: evaluate all N one-step Jacobians in a single compiled call.
    # Solvers need the whole tape of Jacobians every iteration, and calling the scalar
    # version N times dominates the runtime, so map them instead:
    A_map = A.map(N)
    B_map = B.map(N)

    def step_jac_all(Zv, Uv):
        # Zv: (N, nx) states along the trajectory, Uv: flat control tape
        Zc = np.asarray(Zv, dtype=float)[:N, :].T          # (nx, N)
        Uc = np.asarray(Uv, dtype=float).reshape(N, nu).T  # (nu, N)
        Aa = np.asarray(A_map(Zc, Uc))                     # (nx, nx*N)
        Bb = np.asarray(B_map(Zc, Uc))                     # (nx, nu*N)
        As = [Aa[:, k * nx:(k + 1) * nx] for k in range(N)]
        Bs = [Bb[:, k * nu:(k + 1) * nu] for k in range(N)]
        return As, Bs

    # Build the full rollout with a compiled scan (mapaccum) rather than an
    # unrolled Python loop; the unrolled graph is what made high-dimensional
    # nonlinear rollouts (e.g. the 6DOF aircraft) slow to evaluate and compile:
    U = ca.MX.sym("U", N * nu)
    Useq = ca.reshape(U, nu, N)                 # column k is the k-th control
    # The initial state is a symbolic INPUT, not a baked-in constant, so one compiled
    # graph can be re-anchored at any state at runtime (needed for receding-horizon
    # replanning: the plan must start from where the vehicle actually is):
    Z0 = ca.MX.sym("Z0", nx)
    acc = step.mapaccum("roll", N)
    Zc = acc(Z0, Useq)
    Zmat = ca.horzcat(Z0, Zc).T                 # full trajectory (N+1, nx)

    # Select the endpoint components the shoot will constrain:
    tidx = list(range(nx)) if target_idx is None else list(target_idx)
    gend = Zmat[-1, tidx].T

    # Compilation options, optionally with JIT for steady-state speed:
    opts = {"jit": jit, "compiler": "shell",
            "jit_options": {"flags": ["-O2"]}} if jit else {}

    # Compile the endpoint map with its analytic Jacobian and the full rollout:
    F_end = ca.Function("F_end", [U, Z0], [gend, ca.jacobian(gend, U)], opts)
    F_roll = ca.Function("F_roll", [U, Z0], [Zmat], opts)

    # Optionally compile the position Jacobian for obstacle avoidance:
    pos_jac = None
    if pos_idx is not None:
        pos = Zmat[:, list(pos_idx)]            # position rows (N+1, 2)
        # ca.reshape is column-major, so reshape the TRANSPOSE to get row-major order
        # (node0_x, node0_y, node1_x, ...) matching the numpy reshape below:
        posflat = ca.reshape(pos.T, (N + 1) * 2, 1)
        Jpos = ca.jacobian(posflat, U)          # ((N+1)*2, N*nu)
        F_pos = ca.Function("F_pos", [U, Z0], [Jpos], opts)

        # Reshape the compiled position Jacobian to (N+1, 2, N*nu):
        def pos_jac(Uv, z_start=None):
            zs = system.z0 if z_start is None else z_start
            return np.array(F_pos(np.asarray(Uv).flatten(),
                                  np.asarray(zs, float))).reshape(N + 1, 2, N * nu)

    # Assemble the System:
    system = System(F_end, F_roll, step, step_jac, nx, nu, N, z0, dt,
                    target_idx=tidx, pos_jac=pos_jac, job=job, pos_idx=pos_idx)
    system.step_jac_all = step_jac_all

    # Return the ready System:
    return system