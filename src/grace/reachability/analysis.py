# Import package:
import numpy as np


# Compute Controllability Gramian:
def gramian(system, U, reg=0.0):

    # Fetch endpoint Jacobian:
    _, Co = system.endpoint_jac(U)

    # Return Gramian:
    return Co @ Co.T + reg * np.eye(system.m)

# Compute eigenvalues and eigenvectors of system:
def eig(system, U):

    # Fetch Gramian:
    W = gramian(system, U)

    # Compute eigenvalues and eigenvectors:
    vals, vecs = np.linalg.eigh(W)

    # Sort from strongest to weakest:
    order = np.argsort(vals)[::-1]

    # Return eigenvalues and eigenvectors:
    return vals[order], vecs[:, order]

# Minimum control energy to move the endpoint a unit in each principal direction:
def energy_per_direction(system, U, reg=1e-12):

    # Fetch eigenvalues and eigenvectors:
    vals, vecs = eig(system, U)

    # Return energy (J = d^T W^-1 d = 1/lambda for d = 1):
    return 1.0 / (vals + reg), vecs

# Controllability ellipsoid semi-axes (gives an idea on reachability):
def ellipsoid(system, U):

    # Compute eigenvalues and eigenvectors:
    vals, vecs = eig(system, U)

    # Return principle axes and their lengths:
    return np.sqrt(np.clip(vals, 0, None)), vecs

# Condition number of the Gramian:
def condition_number(system, U):

    # Compute eigenvalues:
    vals, _ = eig(system, U)

    # Compute condition number and return:
    return float(vals[0] / vals[-1])

# A readable controllability report for benchmarking a system:
def summary(system, U):

    # Collect the eigenvalues and eigenvectors:
    vals, vecs = eig(system, U)

    # Fetch some common measures:
    ms =  dict(min_eig=float(vals[-1]),
                max_eig=float(vals[0]),
                trace=float(np.sum(vals)),
                log_det=float(np.sum(np.log(vals))),
                condition_number=float(vals[0] / vals[-1]))

    # Fetch energy per direction:
    energy, _ = energy_per_direction(system, U)

    # Package report:
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

    # Return the report:
    return r
