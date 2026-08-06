# Import packages:
import os
import hashlib
import shutil
import subprocess
import numpy as np
import casadi as ca
from .system import System, build, DATA_DIR

# Hash the symbolic graph and dimensions into a short cache key:
def _graph_key(system):

    # Serealize graph and encode:
    graph = system.step.serialize()
    if isinstance(graph, str):
        graph = graph.encode()

    # Dimensions and index sets change the compiled code, so they are keyed too:
    shape = repr((system.nx, system.nu, system.N, float(system.dt),
                  tuple(system.tidx), int(getattr(system, "substeps", 1))))
    return hashlib.sha256(graph + b"|" + shape.encode()).hexdigest()[:16]


# Return the directory that holds a given job's compiled graph:
def _job_dir(job, data_dir):

    # Each job gets its own subdirectory under the data directory:
    return os.path.join(data_dir, job)


# Return the paths of the shared object and metadata for a job:
def _paths(job, data_dir):

    # One shared object holds every compiled function for the job:
    d = _job_dir(job, data_dir)
    return d, os.path.join(d, "graph.so"), os.path.join(d, "meta.npz")

# Check whether a compiled shared object exists for a job and matches the key:
def has_cache(job, data_dir=DATA_DIR, key=None):

    # Both the shared object and its metadata have to be present:
    _, so_path, meta_path = _paths(job, data_dir)
    if not (os.path.exists(so_path) and os.path.exists(meta_path)):
        return False

    # A cache whose key does not match the current graph is stale, not usable:
    try:
        meta = np.load(meta_path, allow_pickle=True)
        if key is None:
            return True
        return str(meta["key"]) == key
    except Exception:
        return False

# Return the C compiler to use, or None if none is available:
def _compiler():

    # Prefer gcc, fall back to clang or cc, and report absence to the caller:
    for name in ("gcc", "clang", "cc"):
        path = shutil.which(name)
        if path is not None:
            return path
    return None


# Build the full set of functions that a reloaded System needs:
def _graph_functions(step, F_roll, F_end, nx, nu, N):

    # Set symbolic graph:
    z = ca.MX.sym("z", nx)
    u = ca.MX.sym("u", nu)
    z_next = step(z, u)

    # One-step Jacobians, and their batched maps over the whole tape:
    A = ca.Function("A", [z, u], [ca.jacobian(z_next, z)])
    B = ca.Function("B", [z, u], [ca.jacobian(z_next, u)])

    # The maps are rewrapped so the emitted symbols have predictable names,
    # since Function.map names the result itself:
    Z_tape = ca.MX.sym("Z_tape", nx, N)
    U_tape = ca.MX.sym("U_tape", nu, N)
    A_map = ca.Function("A_map", [Z_tape, U_tape], [A.map(N)(Z_tape, U_tape)])
    B_map = ca.Function("B_map", [Z_tape, U_tape], [B.map(N)(Z_tape, U_tape)])
    # The state-trajectory Jacobian is emitted too.  A reloaded F_roll is a
    # compiled external and cannot be differentiated, so anything needing dZ/dU
    # has to find it already built:
    U = ca.MX.sym("U", N * nu)
    Z0 = ca.MX.sym("Z0", nx)
    Zs = F_roll(U, Z0)
    flat = ca.reshape(Zs.T, (N + 1) * nx, 1)
    F_stateJ = ca.Function("F_stateJ", [U, Z0], [ca.jacobian(flat, U)])
    funcs = [ca.Function("step", [z, u], [z_next]), F_roll, F_end,
             A, B, A_map, B_map, F_stateJ]

    return funcs


# Emit the graph as C and compile it to a shared object:
def compile_graph(system, key, data_dir=DATA_DIR, opt="-O2", verbose=False):

    # A job name is required to key the compiled files:
    if system.job is None:
        raise ValueError("System has no job name -- pass job= to build() to enable caching")

    # A compiler is required, and its absence is a reason to skip the cache
    # rather than to fail the run:
    cc = _compiler()
    if cc is None:
        return None

    # Ensure the job directory exists:
    d, so_path, meta_path = _paths(system.job, data_dir)
    os.makedirs(d, exist_ok=True)

    # Emit every function the reloaded system needs into a single C file:
    funcs = _graph_functions(system.step, system.F_roll, system.F_end,
                             system.nx, system.nu, system.N)
    c_path = os.path.join(d, "graph.c")
    gen = ca.CodeGenerator(os.path.basename(c_path), {"with_header": False})
    for f in funcs:
        gen.add(f)
    gen.generate(d + os.sep)

    # Compile to a position-independent shared object:
    cmd = [cc, opt, "-fPIC", "-shared", c_path, "-o", so_path]
    if verbose:
        print("[grace] compiling cached graph:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    # A compile failure is reported and the cache is skipped, so the run can
    # still proceed on the uncompiled graph:
    if result.returncode != 0:
        print("[grace] warning: cached-graph compile failed, continuing without cache")
        print(result.stderr.strip()[:500])
        return None

    # The C source is large and no longer needed once the object exists:
    os.remove(c_path)

    # Record the key and the dimensions needed to reassemble the System:
    meta = dict(key=key, nx=system.nx, nu=system.nu, N=system.N, z0=system.z0,
                dt=system.dt, tidx=np.array(system.tidx))
    np.savez(meta_path, **meta)
    return so_path


# Compile a System's graph and record it under its job name:
def save(system, data_dir=DATA_DIR):
    path = compile_graph(system, _graph_key(system), data_dir)
    if path is None:
        return None
    return os.path.dirname(path)

# Load a compiled System from its shared object:
def load(job, data_dir=DATA_DIR):

    # Read the metadata written alongside the shared object:
    d, so_path, meta_path = _paths(job, data_dir)
    meta = np.load(meta_path, allow_pickle=True)
    nx = int(meta["nx"])
    nu = int(meta["nu"])
    N = int(meta["N"])

    # Every function comes straight out of the shared object, so nothing is
    # constructed or differentiated here and the load is a dlopen:
    step = ca.external("step", so_path)
    F_roll = ca.external("F_roll", so_path)
    F_end = ca.external("F_end", so_path)
    A = ca.external("A", so_path)
    B = ca.external("B", so_path)
    A_map = ca.external("A_map", so_path)
    B_map = ca.external("B_map", so_path)

    # Wrap the one-step Jacobians so callers get plain arrays:
    def step_jac(zv, uv):
        return np.array(A(zv, uv)), np.array(B(zv, uv))

    # Batched Jacobians, which evaluate the whole tape in a single call:
    def step_jac_all(Zv, Uv):
        Zc = np.asarray(Zv, dtype=float)[:N, :].T
        Uc = np.asarray(Uv, dtype=float).reshape(N, nu).T
        Aa = np.asarray(A_map(Zc, Uc))
        Bb = np.asarray(B_map(Zc, Uc))
        As = [Aa[:, k * nx:(k + 1) * nx] for k in range(N)]
        Bs = [Bb[:, k * nu:(k + 1) * nu] for k in range(N)]
        return As, Bs

    # Row Jacobians come from the emitted state Jacobian, sliced to the
    # requested indices and cached on the control tape:
    F_stateJ = ca.external("F_stateJ", so_path)

    def row_jac(U, idx):
        key = tuple(idx)
        Uf = np.asarray(U, float).flatten()
        z0 = np.asarray(reloaded.z0, float)
        hit = reloaded._row_cache.get(key)
        if (hit is not None and hit[0].shape == Uf.shape
                and np.array_equal(hit[0], Uf) and np.array_equal(hit[2], z0)):
            return hit[1]
        full = np.array(F_stateJ(Uf, z0)).reshape(N + 1, nx, N * nu)
        J = full[:, list(key), :]
        reloaded._row_cache[key] = (Uf.copy(), J, z0.copy())
        return J

    # Assemble and return the reloaded System:
    reloaded = System(F_end, F_roll, step, step_jac, nx, nu, N,
                      meta["z0"], float(meta["dt"]), target_idx=list(meta["tidx"]),
                      job=job)
    reloaded.step_jac_all = step_jac_all
    reloaded.row_jac = row_jac
    return reloaded

# Build a System, compiling it once and reloading the compiled object after:
def build_cached(dynamics, nx, nu, N, z0, dt, job, target_idx=None,
                 substeps=1, jit=False, data_dir=DATA_DIR, rebuild=False, verbose=False):

    # Build symbolic graph:
    system = build(dynamics, nx, nu, N, z0, dt, target_idx=target_idx,
                   substeps=substeps, jit=False, job=job)
    system.substeps = substeps
    key = _graph_key(system)

    # Compile if there is no matching object, then reload it either way:
    if rebuild or not has_cache(job, data_dir, key):
        if verbose:
            print("[grace] no cached graph for job '%s' (key %s) -- compiling" % (job, key))
        if compile_graph(system, key, data_dir, verbose=verbose) is None:

            # Compilation was unavailable or failed, so hand back the
            # uncompiled system rather than nothing:
            return system
    elif verbose:
        print("[grace] reusing cached graph for job '%s' (key %s)" % (job, key))
    return load(job, data_dir)