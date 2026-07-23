# ============================================================================
# analysis.py -- controllability and reachability analysis:
# ============================================================================
# The controllability Gramian W = Co Co^T of the endpoint map is the object
# that governs reachability: its eigenvectors are the principal reach
# directions and its eigenvalues say how cheaply the endpoint moves in each.
# These tools extract that structure so a system can be benchmarked for how
# controllable it is and where its weak directions lie.
# ============================================================================

import numpy as np


# Controllability Gramian of the endpoint map at a control tape:
def gramian(system, U, reg=0.0):

    # W = Co Co^T where Co = dg/dU is the endpoint Jacobian:
    _, Co = system.endpoint_jac(U)
    return Co @ Co.T + reg * np.eye(system.m)


# Eigenvalues and eigenvectors of the Gramian, sorted strong to weak:
def eig(system, U):

    # Symmetric eigendecomposition of the Gramian:
    W = gramian(system, U)
    vals, vecs = np.linalg.eigh(W)

    # Sort descending so the strongest reach direction comes first:
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


# Minimum control energy to move the endpoint a unit in each principal direction:
def energy_per_direction(system, U, reg=1e-12):

    # The minimum energy to reach a unit endpoint move d is d^T W^-1 d, which
    # for a principal direction is 1 / eigenvalue.  Small eigenvalue means the
    # direction is expensive (weakly controllable):
    vals, vecs = eig(system, U)
    return 1.0 / (vals + reg), vecs


# Controllability ellipsoid semi-axes (reach for a unit control budget):
def ellipsoid(system, U):

    # The reachable set for ||U|| <= 1 is an ellipsoid whose semi-axis lengths
    # are the square roots of the Gramian eigenvalues, oriented by the
    # eigenvectors:
    vals, vecs = eig(system, U)
    return np.sqrt(np.clip(vals, 0, None)), vecs


# Condition number of the Gramian (anisotropy of reachability):
def condition_number(system, U):

    # Ratio of strongest to weakest reach; large means highly anisotropic and
    # fold-prone:
    vals, _ = eig(system, U)
    return float(vals[0] / max(vals[-1], 1e-300))


# Scalar controllability measures collected together:
def measures(system, U):

    # Eigenvalues of the Gramian:
    vals, _ = eig(system, U)

    # Common reachability scalars derived from the spectrum:
    return dict(min_eig=float(vals[-1]),               # weakest controllable direction
                max_eig=float(vals[0]),                # strongest controllable direction
                trace=float(np.sum(vals)),             # total reachability energy
                log_det=float(np.sum(np.log(vals + 1e-300))),  # reachable volume
                condition_number=float(vals[0] / max(vals[-1], 1e-300)))


# A readable controllability report for benchmarking a system:
def summary(system, U):

    # Collect the spectrum and the derived measures:
    vals, vecs = eig(system, U)
    ms = measures(system, U)
    energy, _ = energy_per_direction(system, U)

    # Assemble a structured report:
    report = dict(
        eigenvalues=vals,
        eigenvectors=vecs,
        energy_per_direction=energy,
        measures=ms,
        weakest_direction=vecs[:, -1],
        strongest_direction=vecs[:, 0])

    # Return the report:
    return report


# Print the controllability report in a readable block:
def print_summary(system, U, name="system"):

    # Gather the report:
    r = summary(system, U)
    ms = r["measures"]

    # Print a titled block of the key metrics:
    print(f"=== reachability report: {name} ===")
    print(f"  endpoint dimension        : {system.m}")
    print(f"  strongest reach eigenvalue: {ms['max_eig']:.4e}")
    print(f"  weakest   reach eigenvalue: {ms['min_eig']:.4e}")
    print(f"  reachable volume (log det): {ms['log_det']:.4f}")
    print(f"  anisotropy (condition no.): {ms['condition_number']:.4e}")

    # Report the energy cost per principal direction:
    print("  energy to move endpoint one unit per principal direction:")
    for i, e in enumerate(r["energy_per_direction"]):
        print(f"    direction {i}: {e:.4e}")

    # Return the report for programmatic use:
    return r
