# ============================================================================
# cache.py -- save and reload compiled rollout graphs by job name:
# ============================================================================
# Compiling the CasADi functions is the slow part of building a System.  To
# avoid paying it on every run, a compiled System is serialized to the data
# directory under a job name (for example "cart_pole").  Later runs with the
# same job name reload the compiled graph instead of rebuilding it.
# ============================================================================

import os
import numpy as np
import casadi as ca

from .system import System, build, DATA_DIR


# Return the directory that holds a given job's cached graph:
def _job_dir(job, data_dir):

    # Each job gets its own subdirectory under the data directory:
    return os.path.join(data_dir, job)


# Check whether a compiled graph already exists for a job:
def has_cache(job, data_dir=DATA_DIR):

    # The graph is cached if its meta file is present:
    meta_path = os.path.join(_job_dir(job, data_dir), "meta.npz")
    if not os.path.exists(meta_path):
        return False
    # Graphs saved before the initial state became a symbolic input are incompatible,
    # so treat a cache without the version marker as absent and rebuild it:
    try:
        m = np.load(meta_path, allow_pickle=True)
        return "graph_version" in m.files and int(m["graph_version"]) >= 2
    except Exception:
        return False


# Save a compiled System's graph to the data directory:
def save(system, data_dir=DATA_DIR):

    # A job name is required to key the saved files:
    if system.job is None:
        raise ValueError("System has no job name -- pass job= to build() to enable caching")

    # Ensure the job directory exists:
    d = _job_dir(system.job, data_dir)
    os.makedirs(d, exist_ok=True)

    # Serialize each compiled CasADi function to its own file:
    system.F_end.save(os.path.join(d, "F_end.casadi"))
    system.F_roll.save(os.path.join(d, "F_roll.casadi"))
    system.step.save(os.path.join(d, "step.casadi"))

    # Save the small dimension metadata alongside the graphs.  The position
    # indices are saved too so the position Jacobian can be rebuilt on reload:
    meta = dict(graph_version=2, nx=system.nx, nu=system.nu, N=system.N, z0=system.z0,
                dt=system.dt, tidx=np.array(system.tidx),
                has_pos=system.pos_jac is not None)
    if getattr(system, "pos_idx", None) is not None:
        meta["pos_idx"] = np.array(system.pos_idx)
    np.savez(os.path.join(d, "meta.npz"), **meta)

    # Return the directory the graph was written to:
    return d


# Load a compiled System's graph from the data directory:
def load(job, data_dir=DATA_DIR):

    # Read the metadata:
    d = _job_dir(job, data_dir)
    meta = np.load(os.path.join(d, "meta.npz"))

    # Reload each compiled CasADi function:
    F_end = ca.Function.load(os.path.join(d, "F_end.casadi"))
    F_roll = ca.Function.load(os.path.join(d, "F_roll.casadi"))
    step = ca.Function.load(os.path.join(d, "step.casadi"))

    # Rebuild the one-step Jacobians symbolically from the reloaded step:
    nx = int(meta["nx"])
    nu = int(meta["nu"])
    zs = ca.MX.sym("z", nx)
    us = ca.MX.sym("u", nu)
    znext = step(zs, us)
    A = ca.Function("A", [zs, us], [ca.jacobian(znext, zs)])
    B = ca.Function("B", [zs, us], [ca.jacobian(znext, us)])

    # Wrap the reloaded Jacobians so callers get plain arrays:
    def step_jac(zv, uv):
        return np.array(A(zv, uv)), np.array(B(zv, uv))

    # Rebuild the batched Jacobian map too, so reloaded systems keep the fast path
    # that evaluates the whole tape of one-step Jacobians in a single call:
    Nc = int(meta["N"])
    A_map = A.map(Nc)
    B_map = B.map(Nc)

    def step_jac_all(Zv, Uv):
        Zc = np.asarray(Zv, dtype=float)[:Nc, :].T
        Uc = np.asarray(Uv, dtype=float).reshape(Nc, nu).T
        Aa = np.asarray(A_map(Zc, Uc))
        Bb = np.asarray(B_map(Zc, Uc))
        As = [Aa[:, k * nx:(k + 1) * nx] for k in range(Nc)]
        Bs = [Bb[:, k * nu:(k + 1) * nu] for k in range(Nc)]
        return As, Bs

    # Rebuild the position Jacobian from the reloaded rollout if the system had one.
    # The saved meta records which position indices were used:
    pos_jac = None
    if bool(meta["has_pos"]) and "pos_idx" in meta.files:
        N = int(meta["N"])
        pidx = list(meta["pos_idx"])
        Usym = ca.MX.sym("U", N * nu)
        Z0sym = ca.MX.sym("Z0", nx)
        Zsym = F_roll(Usym, Z0sym)              # (N+1, nx)
        possym = Zsym[:, pidx]                   # (N+1, 2)
        # ca.reshape is column-major, so reshape the TRANSPOSE for row-major order:
        posflat = ca.reshape(possym.T, (N + 1) * 2, 1)
        F_pos = ca.Function("F_pos", [Usym, Z0sym], [ca.jacobian(posflat, Usym)])

        # Reshape the reloaded position Jacobian to (N+1, 2, N*nu):
        def pos_jac(Uv, z_start=None):
            zs = reloaded.z0 if z_start is None else z_start
            return np.array(F_pos(np.asarray(Uv).flatten(),
                                  np.asarray(zs, float))).reshape(N + 1, 2, N * nu)

    # Assemble and return the reloaded System:
    reloaded = System(F_end, F_roll, step, step_jac, nx, nu, int(meta["N"]),
                  meta["z0"], float(meta["dt"]), target_idx=list(meta["tidx"]),
                  pos_jac=pos_jac, job=job)
    reloaded.step_jac_all = step_jac_all
    return reloaded


# Build a System, using the cache when possible:
def build_cached(dynamics, nx, nu, N, z0, dt, job, target_idx=None, pos_idx=None,
                 substeps=1, jit=False, data_dir=DATA_DIR, rebuild=False):

    # Reload the compiled graph if one exists and a rebuild was not requested:
    if has_cache(job, data_dir) and not rebuild:
        return load(job, data_dir)

    # Otherwise compile from the dynamics and save the graph for next time:
    system = build(dynamics, nx, nu, N, z0, dt, target_idx=target_idx,
                   pos_idx=pos_idx, substeps=substeps, jit=jit, job=job)
    save(system, data_dir)
    return system