# Import packages:
import numpy as np
import casadi as ca

# Define aircraft constants:
G = 9.81; RHO = 1.225; MASS = 6.47
IXX, IYY, IZZ = 1.94, 3.88, 5.18
S, B, C = 0.825, 3.0, 0.275; THRUST = 4.83
CL0, CLA = 0.25, 5.0; CD0, KDR = 0.03, 0.04; CYB = -0.3
CM0, CMA, CMDE, CMQ = 0.05, -0.8, -1.2, -22.41
CLB0, CLP, CLDA = -0.05, -0.731, 0.075
CNB, CNR, CNDR = 0.08, -0.060, 0.10

# Define trim:
V_TRIM = 15.0; A_TRIM = np.deg2rad(3.5)
X0 = np.array([V_TRIM * np.cos(A_TRIM), 0, V_TRIM * np.sin(A_TRIM),
               0, 0, 0, 0, A_TRIM, 0, 0, 0, 0])

# Core dynamics with the dihedral-effect derivative Clb as an argument:
def _f(z, u, Clb):

    # Unpack the state and controls:
    uu, vv, ww = z[0], z[1], z[2]; p, q, r = z[3], z[4], z[5]
    phi, theta, psi = z[6], z[7], z[8]; da, de, dr = u[0], u[1], u[2]

    # Airspeed, angle of attack, and sideslip:
    V = ca.sqrt(uu**2 + vv**2 + ww**2 + 1e-6)
    alpha = ca.atan2(ww, uu); beta = ca.asin(ca.fmax(ca.fmin(vv / V, 0.999), -0.999))
    qd = 0.5 * RHO * V**2

    # Aerodynamic forces (lift, drag, side force) in the body frame:
    CL = CL0 + CLA * alpha; CD = CD0 + KDR * CL**2
    L = qd * S * CL; D = qd * S * CD; Yf = qd * S * CYB * beta
    car, sar = ca.cos(alpha), ca.sin(alpha)
    fx_a = L * sar - D * car; fz_a = -L * car - D * sar; fy_a = Yf

    # Aerodynamic moment coefficients (roll uses the passed-in Clb):
    Cl = Clb * beta + CLDA * (da * ca.pi / 180) + CLP * p * B / (2 * V)
    Cm = CM0 + CMA * alpha + CMDE * (de * ca.pi / 180) + CMQ * q * C / (2 * V)
    Cn = CNB * beta + CNDR * (dr * ca.pi / 180) + CNR * r * B / (2 * V)
    Mx = qd * S * B * Cl; My = qd * S * C * Cm; Mz = qd * S * B * Cn

    # Total forces including thrust and gravity:
    fxt = fx_a + THRUST - MASS * G * ca.sin(theta)
    fyt = fy_a + MASS * G * ca.cos(theta) * ca.sin(phi)
    fzt = fz_a + MASS * G * ca.cos(theta) * ca.cos(phi)

    # Body-frame translational accelerations:
    ud = (r * vv - q * ww) + fxt / MASS
    vd = (p * ww - r * uu) + fyt / MASS
    wd = (q * uu - p * vv) + fzt / MASS

    # Body-frame angular accelerations:
    pd = (Mx + (IYY - IZZ) * q * r) / IXX
    qd_ = (My + (IZZ - IXX) * p * r) / IYY
    rd = (Mz + (IXX - IYY) * p * q) / IZZ

    # Euler-angle kinematics:
    cp, sp = ca.cos(phi), ca.sin(phi); ct, st = ca.cos(theta), ca.sin(theta); cy, sy = ca.cos(psi), ca.sin(psi)
    phid = p + (q * sp + r * cp) * ca.tan(theta); thd = q * cp - r * sp
    psid = (q * sp + r * cp) / ca.fmax(ca.fabs(ct), 0.05)

    # Position kinematics (body -> inertial):
    xd = (ct * cy) * uu + (sp * st * cy - cp * sy) * vv + (cp * st * cy + sp * sy) * ww
    yd = (ct * sy) * uu + (sp * st * sy + cp * cy) * vv + (cp * st * sy - sp * cy) * ww
    zd = -st * uu + sp * ct * vv + cp * ct * ww

    # Assemble the state derivative:
    return ca.vertcat(ud, vd, wd, pd, qd_, rd, phid, thd, psid, xd, yd, zd)

# Plain dynamics f(x, u) for shooting and obstacle avoidance (baseline dihedral):
def f(z, u):
    return _f(z, u, CLB0)

# Codesign dynamics, with p being the dihedral effect Clb:
def f_codesign(z, u, p):
    return _f(z, u, p)
