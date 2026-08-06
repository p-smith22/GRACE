# Import packages:
import os
import numpy as np
import casadi as ca


# Default directory for cached rollout graphs:
DATA_DIR = "data"


class System:

    # Hold the compiled functions and problem dimensions for one dynamics model:
    def __init__(self, F_end, F_roll, step, step_jac, nx, nu, N, z0, dt,
                 target_idx=None, job=None):

        # Store the compiled CasADi functions:
        self.F_end = F_end                      # endpoint map U -> [g, dg/dU]
        self.F_roll = F_roll                    # full rollout U -> Z (N+1, nx)
        self.step = step                        # one-step dynamics z, u -> z_next
        self.step_jac = step_jac                # one-step Jacobians z, u -> [A, B]
        self.step_jac_all = None                # batched Jacobians over the whole tape

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

        # Last-value caches for the rollout and endpoint+Jacobian, keyed on the
        # control tape so repeated evaluations on the same U are free.  The
        # solve leans on these heavily: a line search evaluates constraints at a
        # trial control and then, if it accepts, evaluates them again as the new
        # iterate, and without the caches every one of those is a fresh graph
        # call:
        self._roll_cache = None
        self._endjac_cache = None

        # Row Jacobians are compiled per index set on demand and cached the same
        # way, so a limit on one state costs one row rather than all nx:
        self._row_funcs = {}
        self._row_cache = {}

        # Expanding a rollout Jacobian from MX to SX makes it roughly thirty
        # times cheaper to evaluate but costs far more to build, so it is done
        # lazily: a function that has been called enough times to repay the
        # build gets expanded, and a one-shot solve never pays for it:
        self._row_calls = {}
        self._expand_after = 30

    # Roll the control tape out to the full state trajectory:
    def rollout(self, U):

        # Reshape control, extract initial state:
        Uf = np.asarray(U, float).flatten()
        z0 = np.asarray(self.z0, float)

        # Check if rollout was already done and load if so:
        hit = self._roll_cache
        if (hit is not None and hit[0].shape == Uf.shape
                and np.array_equal(hit[0], Uf) and np.array_equal(hit[2], z0)):
            return hit[1]

        # Evaluate the compiled rollout, cache it, and return a plain array if not:
        Z = np.array(self.F_roll(Uf, z0))
        self._roll_cache = (Uf.copy(), Z, z0.copy())
        return Z

    # Evaluate the endpoint and its analytic Jacobian at a control sequence:
    def endpoint_jac(self, U):

        # Reshape control, extract initial state:
        Uf = np.asarray(U, float).flatten()
        z0 = np.asarray(self.z0, float)

        # Check if rollout was already done and load if so:
        hit = self._endjac_cache
        if (hit is not None and hit[0].shape == Uf.shape
                and np.array_equal(hit[0], Uf) and np.array_equal(hit[3], z0)):
            return hit[1], hit[2]

        # Evaluate the compiled endpoint map, cache, and return plain arrays if not:
        g, J = self.F_end(Uf, z0)
        g = np.array(g).flatten()
        J = np.array(J)
        self._endjac_cache = (Uf.copy(), g, J, z0.copy())
        return g, J

    # Evaluate the endpoint only:
    def endpoint(self, U):

        # Reuse endpoint_jac function, but just return endpoint:
        g, _ = self.endpoint_jac(U)
        return g

    # Jacobian of a chosen set of state rows along the whole trajectory:
    def row_jac(self, U, idx):

        # A limit on one state should cost one row, not all nx of them, so a
        # small function is compiled per index set the first time it is asked
        # for and then cached like everything else:
        key = tuple(idx)
        Uf = np.asarray(U, float).flatten()
        z0 = np.asarray(self.z0, float)

        # Return the cached rows if this is the same tape and start:
        hit = self._row_cache.get(key)
        if (hit is not None and hit[0].shape == Uf.shape
                and np.array_equal(hit[0], Uf) and np.array_equal(hit[2], z0)):
            return hit[1]

        # Compile the row Jacobian for this index set once:
        if key not in self._row_funcs:
            Us = ca.MX.sym("U", self.N * self.nu)
            Z0 = ca.MX.sym("Z0", self.nx)
            Zs = self.F_roll(Us, Z0)
            flat = ca.reshape(Zs[:, list(key)].T, (self.N + 1) * len(key), 1)
            self._row_funcs[key] = ca.Function("F_row", [Us, Z0],
                                               [ca.jacobian(flat, Us)])

        # Once this index set has been asked for often enough, the expansion
        # pays for itself over the calls still to come:
        n = self._row_calls.get(key, 0) + 1
        self._row_calls[key] = n
        if n == self._expand_after:
            try:
                self._row_funcs[key] = self._row_funcs[key].expand()
            except Exception:
                self._expand_after = 10 ** 9

        # Evaluate, cache and return:
        J = np.array(self._row_funcs[key](Uf, z0)).reshape(
            self.N + 1, len(key), self.N * self.nu)
        self._row_cache[key] = (Uf.copy(), J, z0.copy())
        return J

    # Reduce a full state target to the constrained components:
    def target(self, z_target):

        # Fetch endpoint:
        zt = np.asarray(z_target, float)

        # Return only if state was identified as constrained:
        return zt[self.tidx] if len(zt) == self.nx else zt

    # Evaluate the numeric one-step dynamics:
    def step_np(self, z, u):

        # Evaluate the compiled step and return a plain array:
        return np.array(self.step(z, u)).flatten()

# Build a compiled System from a user's dynamics function:
def build(dynamics, nx, nu, N, z0, dt, target_idx=None,
          substeps=1, jit=True, job=None, data_dir=DATA_DIR, rebuild=False):

    # Symbolic definitions:
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    U = ca.MX.sym("U", N * nu)
    Useq = ca.reshape(U, nu, N)


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

    # Compile matrices across whole trajectory:
    A_map = A.map(N)
    B_map = B.map(N)

    # Compute jacobians across entire trajectory for shooting solvers:
    def step_jac_all(Zv, Uv):
        Zc = np.asarray(Zv, dtype=float)[:N, :].T
        Uc = np.asarray(Uv, dtype=float).reshape(N, nu).T
        Aa = np.asarray(A_map(Zc, Uc))
        Bb = np.asarray(B_map(Zc, Uc))
        As = [Aa[:, k * nx:(k + 1) * nx] for k in range(N)]
        Bs = [Bb[:, k * nu:(k + 1) * nu] for k in range(N)]
        return As, Bs

    # Accumulate steps:
    Z0 = ca.MX.sym("Z0", nx)
    acc = step.mapaccum("roll", N)
    Zc = acc(Z0, Useq)
    Zmat = ca.horzcat(Z0, Zc).T

    # Select the endpoint components the shoot will constrain:
    tidx = list(range(nx)) if target_idx is None else list(target_idx)
    gend = Zmat[-1, tidx].T

    # Compilation options, optionally with JIT for steady-state speed:
    opts = {"jit": jit, "compiler": "shell",
            "jit_options": {"flags": ["-O2"]}} if jit else {}

    # Compile the endpoint map with its analytic Jacobian and the full rollout:
    F_end = ca.Function("F_end", [U, Z0], [gend, ca.jacobian(gend, U)], opts)
    F_roll = ca.Function("F_roll", [U, Z0], [Zmat], opts)

    # Assemble the System:
    system = System(F_end, F_roll, step, step_jac, nx, nu, N, z0, dt,
                    target_idx=tidx, job=job)
    system.step_jac_all = step_jac_all

    # Return the ready System:
    return system