"""
Functions for evaluating electrostatics between atom-centered multipoles
"""

from importlib import resources
import pandas as pd
import numpy as np
from . import constants
import torch
from typing import Tuple
import qcelemental as qcel


def proc_molden(name):
    """Get coordinates (a.u.) and atom types from a molden.
    Accounts for ghost atoms"""

    with open(name, "r") as fp:
        data = fp.read().split("[GTO]")[0].strip()
    data = data.split("\n")[2:]
    data = [line.strip().split() for line in data]
    Z = [line[0] for line in data]

    try:
        Z = [constants.elem_to_z[elem] for elem in Z]
    except:
        print(name)
        return 0, 0
    Z = np.array(Z, dtype=np.int64)

    R = [line[3:] for line in data]
    R = [[float(xyz) for xyz in line] for line in R]
    R = np.array(R, dtype=np.float64)

    mask = [line[2] for line in data]
    mask = [(d != "0") for d in mask]

    return R[mask], Z[mask]


def make_quad_np(flat_quad):
    natom = flat_quad.shape[0]
    full_quad = np.zeros((natom, 3, 3))
    full_quad[:, 0, 0] = flat_quad[:, 0]  # xx
    full_quad[:, 0, 1] = flat_quad[:, 1]  # xy
    full_quad[:, 1, 0] = flat_quad[:, 1]  # xy
    full_quad[:, 0, 2] = flat_quad[:, 2]  # xz
    full_quad[:, 2, 0] = flat_quad[:, 2]  # xz
    full_quad[:, 1, 1] = flat_quad[:, 3]  # yy
    full_quad[:, 1, 2] = flat_quad[:, 4]  # yz
    full_quad[:, 2, 1] = flat_quad[:, 4]  # yz
    full_quad[:, 2, 2] = flat_quad[:, 5]  # zz

    trace = full_quad[:, 0, 0] + full_quad[:, 1, 1] + full_quad[:, 2, 2]

    full_quad[:, 0, 0] -= trace / 3.0
    full_quad[:, 1, 1] -= trace / 3.0
    full_quad[:, 2, 2] -= trace / 3.0

    return full_quad


def qpole_redundant(unique):
    assert len(unique) == 6
    redundant = np.zeros((3, 3))

    redundant[0, 0] = unique[0]
    redundant[0, 1] = unique[1]
    redundant[1, 0] = unique[1]
    redundant[0, 2] = unique[2]
    redundant[2, 0] = unique[2]
    redundant[1, 1] = unique[3]
    redundant[1, 2] = unique[4]
    redundant[2, 1] = unique[4]
    redundant[2, 2] = unique[5]
    return redundant


def ensure_traceless_qpole(qpole):
    # get device of qpole
    qpole_mask = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=qpole.dtype,
        device=qpole.device,
    )
    trace = qpole * qpole_mask
    trace = torch.sum(trace, dim=[1, 2], keepdim=True) / 3.0
    trace = qpole_mask * trace
    qpole = qpole - trace
    return qpole


def qpole_expand_and_traceless(qpole):
    qpole = torch.tensor(qpole_redundant(qpole))
    qpole = ensure_traceless_qpole(qpole)
    return qpole


def ensure_traceless_qpole_torch(qpole):
    qpoles = []
    for i in range(qpole.shape[0]):
        qpoles.append(qpole_expand_and_traceless(qpole[i]).view(3, 3))
    qpole = torch.stack(qpoles)
    qpole = torch.cat((qpole[:, 0, :], qpole[:, 1, 1:], qpole[:, 2, 2:]), dim=1)
    return qpole


def compact_multipoles_to_charge_dipole_qpoles(multipoles):
    charges = multipoles[:, 0]
    dipoles = multipoles[:, 1:4]
    qpoles = multipoles[:, 4:10]
    qpoles_out = []
    for i in range(qpoles.shape[0]):
        # qpoles_out.append(qpole_redundant(qpoles[i]))
        qpoles_out.append(qpole_expand_and_traceless(qpoles[i]))
    qpoles = np.array(qpoles_out)
    return charges, dipoles, qpoles


def charge_dipole_qpoles_to_compact_multipoles(charges, dipoles, qpoles):
    multipoles = np.zeros((charges.shape[0], 10))
    multipoles[:, 0] = charges
    multipoles[:, 1:4] = dipoles
    for i in range(qpoles.shape[0]):
        multipoles[i, 4:10] = qpoles[i].flatten()[[0, 1, 2, 4, 5, 8]]
    return multipoles


def T_cart_Z_MTP(RA, RB, alpha_j=None):
    lam_1, lam_3, lam_5 = (1.0, 1.0, 1.0)

    dR = RB - RA
    R = np.linalg.norm(dR)

    if alpha_j is not None:
        lam_1, lam_3, lam_5 = elst_damping_z_mtp(alpha_j, R)

    # print(f"{lam_1=:.4f}")
    # print(f"{lam_3:.8f}")
    # print(f"{lam_5:.8f}")
    """
[5.23691392 6.1524798  6.15149873],
lam_1=array([0.99997837, 0.99996192, 0.99996186])
lam_3=array([0.99974603, 0.99957443, 0.9995738 ])
lam_5=array([0.99891413, 0.99826007, 0.99825773])

it[:,1] = [-0.03643653 -0.02552135 -0.02549818]
    """

    delta = np.identity(3)

    T0 = R**-1 * lam_1
    # print(f"{R**-1:.4f}, {lam_1=:.4f}, {T0=:.4f}")
    T1 = (R**-3) * (-1.0 * dR) * lam_3
    # print(f"{lam_3 = }")
    # print(f"{T1 = }")
    T2 = (R**-5) * (lam_5 * 3 * np.outer(dR, dR) - lam_3 * R * R * delta)
    return T0, T1, T2


def T_cart(RA, RB, alpha_i=None, alpha_j=None):
    lam_1, lam_3, lam_5, lam_7, lam_9 = (1.0, 1.0, 1.0, 1.0, 1.0)

    dR = RB - RA
    R = np.linalg.norm(dR)
    if alpha_i is not None and alpha_j is not None:
        lam_1, lam_3, lam_5, lam_7, lam_9 = elst_damping_mtp_mtp(alpha_i, alpha_j, R)
    # print(f"{R:.6f}, {lam_1=:.8f}")

    # print(f"{R:.6f},{lam_1=:.8f}, {lam_3=:.8f}\n {lam_5=:.8f}, {lam_7=:.8f}, {lam_9=:.8f}")
    delta = np.identity(3)

    # print(f"{R:.6f},{lam_3=:.8f} {lam_5=:.8f}")
    # E_qq
    T0 = R**-1 * lam_1
    # E_qu
    # print(f"{float(lam_1)=:.4f}")
    # print(f"{float(T0)   =:.4f}")
    # print(f"{T1 = }")
    # print(f"{float(lam_3)=:.4f}")
    T1 = (R**-3) * (-1.0 * dR) * lam_3
    print(f"{T1 = }")
    # E_uu, E_qQ
    T2 = (R**-5) * (lam_5 * 3 * np.outer(dR, dR) - lam_3 * R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    # E_uQ
    # lam_5_const = np.identity(3) * 6 + np.ones((3, 3)) * 3
    lam_5_const = np.ones((3, 3, 3)) * 3
    # lam_5_const[0, 0, 0] *= 3
    # lam_5_const[1, 1, 1] *= 3
    # lam_5_const[2, 2, 2] *= 3
    lam_5_const *= lam_5
    # lam_5_const = np.identity(3) * -6 + np.ones((3, 3)) * 9
    # print(lam_5_const)
    # print(Rdd.shape)
    # print(Rdd)
    # print(Rdd * lam_5_const)
    # print(np.ones_like(Rdd) * lam_5_const)
    # lam_5_const = 1
    """
    undamped T3:
T3:
undamped T3 cliff:
[[-7.95496337e-03 -4.82909142e-04  5.15420225e-05] 
 [-4.82909142e-04  3.96843347e-03  1.95373020e-06] 
 [ 5.15420225e-05  1.95373020e-06  3.98652990e-03] 
 [-4.82909142e-04  3.96843347e-03  1.95373020e-06] 
 [ 3.96843347e-03  3.62047858e-04 -1.28412503e-05] 
 [ 1.95373020e-06 -1.28412503e-05  1.20861284e-04] 
 [ 5.15420225e-05  1.95373020e-06  3.98652990e-03] 
 [ 1.95373020e-06 -1.28412503e-05  1.20861284e-04] 
 [ 3.98652990e-03  1.20861284e-04 -3.87007722e-05]]    

 damped T3 cliff:
[[-7.71632368e-03 -4.74231116e-04  5.06157965e-05] 
 [-4.74231116e-04  3.94491880e-03  1.92331462e-06] 
 [ 5.06157965e-05  1.92331462e-06  3.77140488e-03] 
 [-4.74231116e-04  3.94491880e-03  1.92331462e-06] 
 [ 3.94491880e-03  3.59891867e-04 -1.27651604e-05] 
 [ 1.92331462e-06 -1.27651604e-05  1.14339249e-04] 
 [ 5.06157965e-05  1.92331462e-06  3.77140488e-03] 
 [ 1.92331462e-06 -1.27651604e-05  1.14339249e-04] 
 [ 3.77140488e-03  1.14339249e-04 -3.78506361e-05]]      
    """
    T3 = (
        (R**-7)
        * (
            -15 * lam_7 * np.multiply.outer(np.outer(dR, dR), dR)
            + lam_5_const * R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
        )
    )
    # print("T3:\nq", T3)
    # print("T3:")
    # print(T3)
    # print(-T3.transpose(1, 0, 2))
    # print(-T3.transpose(0, 1, 2))

    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    # Used for E_QQ
    T4 = (R**-9) * (
        105 * lam_9 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - 15 * lam_7 * R * R * (
            RRdd
            + RRdd.transpose(0, 2, 1, 3)
            + RRdd.transpose(0, 3, 2, 1)
            + RRdd.transpose(2, 1, 0, 3)
            + RRdd.transpose(3, 1, 2, 0)
            + RRdd.transpose(2, 3, 0, 1)
        )
        + 3 * lam_5 * (R**4) * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
    )

    return T0, T1, T2, T3, T4


def thole_damping(r_ij, alpha_i, alpha_j, a):
    """Apply Thole damping to interaction tensor"""
    # Compute damping factor
    u = r_ij / ((alpha_i * alpha_j) ** (1.0 / 6.0))
    au3 = a * (u**3)
    l3 = 1 - np.exp(-au3)
    l5 = 1 - (1 + au3) * np.exp(-au3)
    l7 = 1 - (1.0 + au3 + 0.6 * au3**2) * np.exp(-au3)
    l9 = 1 - (1 + au3 + (18 * au3**2 + 9 * au3**3) / 35) * np.exp(-au3)
    return au3, l3, l5, l7, l9

def elst_damping_mtp_mtp(alpha_i, alpha_j, r):
    """
    # MTP-MTP interaction from CLIFF
    # Get the lambdas
    lam1 = 1.0
    lam3 = 1.0
    lam5 = 1.0
    lam7 = 1.0
    lam9 = 1.0

    if abs(alpha1 - alpha2) > 1e-6:
        A = a2_2 / (a2_2 - a1_2)
        B = a1_2 / (a1_2 - a2_2)

        
        lam1 -= A*e1r
        lam1 -= B*e2r

        lam3 -= (1.0 + alpha1*r)*A*e1r 
        lam3 -= (1.0 + alpha2*r)*B*e2r

        lam5 -= (1.0 + alpha1*r + (1.0/3.0)*a1_2*r2)*A*e1r
        lam5 -= (1.0 + alpha2*r + (1.0/3.0)*a2_2*r2)*B*e2r

        lam7 -= (1.0 + alpha1*r + (2.0/5.0)*a1_2*r2 + (1.0/15.0)*a1_3*r3)*A*e1r
        lam7 -= (1.0 + alpha2*r + (2.0/5.0)*a2_2*r2 + (1.0/15.0)*a2_3*r3)*B*e2r

        lam9 -= (1.0 + alpha1*r + (3.0/7.0)*a1_2*r2 + (2.0/21.0)*a1_3*r3 + (1.0/105.0)*a1_4*r4)*A*e1r
        lam9 -= (1.0 + alpha2*r + (3.0/7.0)*a2_2*r2 + (2.0/21.0)*a2_3*r3 + (1.0/105.0)*a2_4*r4)*B*e2r

    else:
        # assume alpha1 == alpha2
    
        lam1 -= (1.0 + 0.5*alpha1*r)*e1r
        lam3 -= (1.0 + alpha1*r + 0.5*a1_2*r2)*e1r
        lam5 -= (1.0 + alpha1*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3)*e1r
        lam7 -= (1.0 + alpha1*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3 + (1.0/30.0)*a1_4*r4)*e1r
        lam9 -= (1.0 + alpha1*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3 + (4.0/105.0)*a1_4*r4 + (1.0/210.0)*a1_4*alpha1*r5)*e1r
    """
    r2 = r**2
    r3 = r2*r
    r4 = r2**2
    r5 = r4*r
    a1_2 = alpha_i*alpha_i
    a2_2 = alpha_j*alpha_j
    a1_3 = a1_2*alpha_i
    a2_3 = a2_2*alpha_j
    a1_4 = a1_3*alpha_i
    a2_4 = a2_3*alpha_j
    e1r = np.exp(-1.0 * alpha_i * r)
    e2r = np.exp(-1.0 * alpha_j * r)
    lam1, lam3, lam5, lam7, lam9 = (1.0, 1.0, 1.0, 1.0, 1.0)
    if abs(alpha_i - alpha_j) > 1e-6:
        A = a2_2 / (a2_2 - a1_2)
        # print(f"{A=:.4f}")
        B = a1_2 / (a1_2 - a2_2)
        lam1 -= A*e1r
        lam1 -= B*e2r
        lam3 -= (1.0 + alpha_i*r)*A*e1r
        lam3 -= (1.0 + alpha_j*r)*B*e2r

        lam5 -= (1.0 + alpha_i*r + (1.0/3.0)*a1_2*r2)*A*e1r
        lam5 -= (1.0 + alpha_j*r + (1.0/3.0)*a2_2*r2)*B*e2r

        lam7 -= (1.0 + alpha_i*r + (2.0/5.0)*a1_2*r2 + (1.0/15.0)*a1_3*r3)*A*e1r
        lam7 -= (1.0 + alpha_j*r + (2.0/5.0)*a2_2*r2 + (1.0/15.0)*a2_3*r3)*B*e2r

        lam9 -= (1.0 + alpha_i*r + (3.0/7.0)*a1_2*r2 + (2.0/21.0)*a1_3*r3 + (1.0/105.0)*a1_4*r4)*A*e1r
        lam9 -= (1.0 + alpha_j*r + (3.0/7.0)*a2_2*r2 + (2.0/21.0)*a2_3*r3 + (1.0/105.0)*a2_4*r4)*B*e2r

    else:
        # print("A = 0.0")
        lam1 -= (1.0 + 0.5*alpha_i*r)*e1r
        lam3 -= (1.0 + alpha_i*r + 0.5*a1_2*r2)*e1r
        lam5 -= (1.0 + alpha_i*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3)*e1r
        lam7 -= (1.0 + alpha_i*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3 + (1.0/30.0)*a1_4*r4)*e1r
        lam9 -= (1.0 + alpha_i*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3 + (4.0/105.0)*a1_4*r4 + (1.0/210.0)*a1_4*alpha_i*r5)*e1r
    return lam1, lam3, lam5, lam7, lam9


def elst_damping_z_mtp(alpha_j, r):
    """
    # Z-MTP interaction from CLIFF
    lam_1 = 1.0 - np.exp(-1.0 * np.multiply(alpha2,r))
    lam_3 = 1.0 - (1.0 + np.multiply(alpha2,r)) * np.exp(-1.0*np.multiply(alpha2,r)) 
    lam_5 = 1.0 - (1.0 + np.multiply(alpha2,r) + (1.0/3.0)*np.multiply(np.square(alpha2),r2)) * np.exp(-1.0*np.multiply(alpha2,r))
    # TODO: remove the damping by setting all lam* = 1.0
    if damping == False:
        lam_1 = 1.0
        lam_3 = 1.0
        lam_5 = 1.0
    """
    # Z-MTP interaction from CLIFF
    lam_1 = 1.0 - np.exp(-1.0 * np.multiply(alpha_j,r))
    lam_3 = 1.0 - (1.0 + np.multiply(alpha_j,r)) * np.exp(-1.0*np.multiply(alpha_j,r)) 
    lam_5 = 1.0 - (1.0 + np.multiply(alpha_j,r) + (1.0/3.0)*np.multiply(np.square(alpha_j), r**2)) * np.exp(-1.0*np.multiply(alpha_j,r))
    return lam_1, lam_3, lam_5


def T_cart_Thole_damping(RA, RB, alpha_i, alpha_j, a):
    dR = RB - RA
    R = np.linalg.norm(dR)

    delta = np.identity(3)

    au3, l3, l5, l7, l9 = thole_damping(R, alpha_i, alpha_j, a)

    T0 = R**-1
    T1 = l3 * (R**-3) * (-1.0 * dR)
    T2 = (R**-5) * (l5 * 3 * np.outer(dR, dR) - l3 * R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    T3 = (
        (R**-7)
        * -1.0
        * (
            l7 * 15 * np.multiply.outer(np.outer(dR, dR), dR)
            - l5 * 3 * R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
        )
    )

    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    T4 = (R**-9) * (
        l9 * 105 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - l7
        * 15
        * R
        * R
        * (
            RRdd
            + RRdd.transpose(0, 2, 1, 3)
            + RRdd.transpose(0, 3, 2, 1)
            + RRdd.transpose(2, 1, 0, 3)
            + RRdd.transpose(3, 1, 2, 0)
            + RRdd.transpose(2, 3, 0, 1)
        )
        + l5
        * 3
        * (R**4)
        * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
    )
    return T0, T1, T2, T3, T4


def T_cart_torch(RA, RB):
    """
    Compute the multipole interaction tensors for N_A x N_B atom pairs.
    Args:
        RA: Tensor of shape (N_A, 3), positions of set A.
        RB: Tensor of shape (N_B, 3), positions of set B.
    Returns:
        T0: (N_A, N_B)
        T1: (N_A, N_B, 3)
        T2: (N_A, N_B, 3, 3)
        T3: (N_A, N_B, 3, 3, 3)
        T4: (N_A, N_B, 3, 3, 3, 3)
    """
    import torch

    # Get dimensions
    N_A = RA.shape[0]
    N_B = RB.shape[0]
    device = RA.device

    # Reshape for broadcasting: RA [N_A, 1, 3], RB [1, N_B, 3]
    RA_expanded = RA.unsqueeze(1)  # [N_A, 1, 3]
    RB_expanded = RB.unsqueeze(0)  # [1, N_B, 3]

    # Compute displacement vectors for all pairs
    dR = RB_expanded - RA_expanded  # [N_A, N_B, 3]

    # Compute distance for all pairs
    R_squared = torch.sum(dR**2, dim=2)  # [N_A, N_B]
    R = torch.sqrt(R_squared)  # [N_A, N_B]

    # Avoid division by zero by adding small epsilon
    eps = 1e-10
    R_safe = torch.clamp(R, min=eps)

    # Identity tensor
    delta = torch.eye(3, device=device)  # [3, 3]

    # T0: Charge-charge interaction tensor [N_A, N_B]
    T0 = 1.0 / R_safe

    # T1: Charge-dipole interaction tensor [N_A, N_B, 3]
    # R^-3 * (-dR)
    R_inv_cubed = 1.0 / (R_safe**3)
    T1 = -dR * R_inv_cubed.unsqueeze(-1)  # [N_A, N_B, 3]

    # T2: Dipole-dipole interaction tensor [N_A, N_B, 3, 3]
    R_inv_fifth = 1.0 / (R_safe**5)

    # Compute outer product of dR with itself for all pairs
    dR_outer = torch.einsum("...i,...j->...ij", dR, dR)  # [N_A, N_B, 3, 3]

    # 3 * (dR ⊗ dR) - R^2 * δ
    T2_term1 = 3.0 * dR_outer  # [N_A, N_B, 3, 3]
    T2_term2 = R_squared.unsqueeze(-1).unsqueeze(-1) * delta  # [N_A, N_B, 3, 3]
    T2 = (T2_term1 - T2_term2) * R_inv_fifth.unsqueeze(-1).unsqueeze(
        -1
    )  # [N_A, N_B, 3, 3]

    # T3: Dipole-quadrupole interaction tensor [N_A, N_B, 3, 3, 3]
    R_inv_seventh = 1.0 / (R_safe**7)

    # Create Rdd tensor: dR_i * δ_jk for all pairs
    Rdd = torch.einsum("...i,jk->...ijk", dR, delta)  # [N_A, N_B, 3, 3, 3]

    # Create dR_i * dR_j * dR_k tensor
    dR_outer_outer = torch.einsum(
        "...i,...j,...k->...ijk", dR, dR, dR
    )  # [N_A, N_B, 3, 3, 3]

    # Calculate T3
    T3_term1 = 15.0 * dR_outer_outer  # [N_A, N_B, 3, 3, 3]

    # Sum of permuted Rdd tensors
    # Rdd has shape [N_A, N_B, 3, 3, 3] with indices [batch_A, batch_B, i, j, k]
    # We want to permute the last 3 dimensions
    Rdd_sum = Rdd + Rdd.permute(0, 1, 3, 2, 4) + Rdd.permute(0, 1, 4, 3, 2)
    T3_term2 = 3.0 * R_squared.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * Rdd_sum

    T3 = (
        -1.0
        * (T3_term1 - T3_term2)
        * R_inv_seventh.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )

    # T4: Quadrupole-quadrupole interaction tensor [N_A, N_B, 3, 3, 3, 3]
    R_inv_ninth = 1.0 / (R_safe**9)

    # Create RRdd tensor: dR_i * dR_j * δ_kl
    # We need to expand delta to match batch dimensions
    delta_expanded = delta.view(1, 1, 3, 3).expand(N_A, N_B, 3, 3)
    RRdd = torch.einsum(
        "...ij,...kl->...ijkl", dR_outer, delta_expanded
    )  # [N_A, N_B, 3, 3, 3, 3]

    # Create δδ tensor: δ_ij * δ_kl
    dddd = torch.einsum("ij,kl->ijkl", delta, delta)
    dddd_expanded = dddd.view(1, 1, 3, 3, 3, 3).expand(N_A, N_B, 3, 3, 3, 3)

    # Create dR_i * dR_j * dR_k * dR_l tensor
    dR_outer_outer_outer = torch.einsum("...ij,...kl->...ijkl", dR_outer, dR_outer)

    # Calculate T4
    T4_term1 = 105.0 * dR_outer_outer_outer

    # Sum of permuted RRdd tensors
    # RRdd has shape [N_A, N_B, 3, 3, 3, 3] with indices [batch_A, batch_B, i, j, k, l]
    # We need to permute the last 4 dimensions: i, j, k, l
    RRdd_sum = (
        RRdd
        + RRdd.permute(0, 1, 2, 4, 3, 5)  # i,k,j,l
        + RRdd.permute(0, 1, 2, 5, 4, 3)  # i,l,k,j
        + RRdd.permute(0, 1, 4, 3, 2, 5)  # k,j,i,l
        + RRdd.permute(0, 1, 5, 3, 4, 2)  # l,j,k,i
        + RRdd.permute(0, 1, 4, 5, 2, 3)  # k,l,i,j
    )

    T4_term2 = (
        15.0
        * R_squared.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        * RRdd_sum
    )

    # Sum of permuted dddd tensors
    dddd_sum = (
        dddd_expanded
        + dddd_expanded.permute(0, 1, 2, 4, 3, 5)  # i,k,j,l
        + dddd_expanded.permute(0, 1, 2, 5, 4, 3)  # i,l,k,j
    )

    T4_term3 = (
        3.0
        * (R_squared**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        * dddd_sum
    )

    T4 = (T4_term1 - T4_term2 + T4_term3) * R_inv_ninth.unsqueeze(-1).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1)

    return T0, T1, T2, T3, T4


def eval_qcel_dimer(mol_dimer, qA, muA, thetaA, qB, muB, thetaB):
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.
    """
    total_energy = 0.0
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]

            pair_energy = eval_interaction(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j
            )
            total_energy += pair_energy
    return total_energy * constants.h2kcalmol


def eval_qcel_dimer_individual(mol_dimer, qA, muA, thetaA, qB, muB, thetaB, match_cliff=False) -> float:
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.
    """
    total_energy = np.zeros(3)
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]

            E_q, E_dp, E_qpole = eval_interaction_individual(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j
            )
            total_energy[0] += E_q
            total_energy[1] += E_dp
            total_energy[2] += E_qpole
    return total_energy * constants.h2kcalmol


def eval_qcel_dimer_individual_components(
    mol_dimer, qA, muA, thetaA, qB, muB, thetaB, alphaA=None, alphaB=None, traceless=True, amoeba_eq=False, match_cliff=True
) -> Tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.

    NOTE: the mu-Q and Q-Q damping terms do not agree with the CLIFF
    implementation. If damping is disabled or only care about the q-q, q-u,
    u-u, and q-Q terms then this function can be used. AP2 only uses these q-q,
    q-u, u-u, and q-Q terms, so this implementation is sufficient for AP2.
    """
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    t = np.zeros((len(ZA), len(ZB)))
    E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs = (
        t.copy(),
        t.copy(),
        t.copy(),
        t.copy(),
        t.copy(),
        t.copy(),
    )
    E_ZA_MBs, E_ZB_MAs, E_ZA_ZBs = (
        t.copy(),
        t.copy(),
        t.copy(),
    )
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]
            a_i = alphaA[i] if alphaA is not None and alphaB is not None else None
            a_j = alphaB[j] if alphaA is not None and alphaB is not None else None
            za_i = ZA[i] if amoeba_eq else None
            zb_j = ZB[j] if amoeba_eq else None
            E_qq, E_qu, E_uu, E_qQ, E_uQ, E_QQ, E_ZA_ZB, E_ZA_MB, E_ZB_MA = eval_interaction_individual_components(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j, ZA=za_i, ZB=zb_j,
                alpha_i=a_i, alpha_j=a_j,
                traceless=traceless,
                match_cliff=match_cliff,
            )
            E_qqs[i, j] = E_qq
            E_qus[i, j] = E_qu
            E_uus[i, j] = E_uu
            E_qQs[i, j] = E_qQ
            E_uQs[i, j] = E_uQ
            E_QQs[i, j] = E_QQ
            if amoeba_eq:
                E_ZA_ZBs[i, j] = E_ZA_ZB
                E_ZA_MBs[i, j] = E_ZA_MB
                E_ZB_MAs[i, j] = E_ZB_MA
    total_energy = (
        np.sum(E_qqs)
        + np.sum(E_qus)
        + np.sum(E_uus)
        + np.sum(E_qQs)
        + np.sum(E_uQs)
        + np.sum(E_QQs)
        + np.sum(E_ZA_MBs)
        + np.sum(E_ZB_MAs)
        + np.sum(E_ZA_ZBs)
    )
    # print(f"{E_ZA_MBs.flatten() }")
    print(f"{E_ZB_MAs.flatten() }")
    print(f"{np.sum(E_ZB_MAs.flatten())}")
    print(f"{np.sum(E_ZB_MAs.flatten()) * 627.5094737775374 }")
    total_energy *= constants.h2kcalmol
    E_qqs *= constants.h2kcalmol
    E_qus *= constants.h2kcalmol
    E_uus *= constants.h2kcalmol
    E_qQs *= constants.h2kcalmol
    E_uQs *= constants.h2kcalmol
    E_QQs *= constants.h2kcalmol
    E_ZA_MBs *= constants.h2kcalmol
    E_ZB_MAs *= constants.h2kcalmol
    E_ZA_ZBs *= constants.h2kcalmol
    return total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs, E_ZA_ZBs, E_ZA_MBs, E_ZB_MAs


def eval_interaction_individual(
    RA, qA, muA, thetaA, RB, qB, muB, thetaB, traceless=False
):
    T0, T1, T2, T3, T4 = T_cart(RA, RB)

    # Most inputs will already be traceless, but we can ensure this is the case
    if not traceless:
        # divide by 3 because of traceless Buckingham quadrupoles have T_ij(2)
        # annihalite the trace
        traceA = np.trace(thetaA)
        thetaA[0, 0] -= traceA / 3.0
        thetaA[1, 1] -= traceA / 3.0
        thetaA[2, 2] -= traceA / 3.0
        traceB = np.trace(thetaB)
        thetaB[0, 0] -= traceB / 3.0
        thetaB[1, 1] -= traceB / 3.0
        thetaB[2, 2] -= traceB / 3.0

    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(
        T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))
    ) * (-1.0 / 3.0)

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * (1.0 / 9.0)
    # partial-charge electrostatic energy
    E_q = E_qq
    # dipole correction
    E_u = E_qu + E_uu
    # quadrupole correction
    E_Q = E_qQ + E_uQ + E_QQ
    return E_q, E_u, E_Q


def eval_interaction_individual_components(
    RA, qA, muA, thetaA, RB, qB, muB, thetaB, ZA=None, ZB=None,
    alpha_i=None, alpha_j=None,
    traceless=False, match_cliff=False
):
    """
    if alpha_i and alpha_j are provided, Thole damping is applied.

    if amoeba_eq is True, evaulate 4 elst terms instead of just MTP-MTP. Note
    the charge term has Z subtracted from qA and qB, so it is not the same as
    the MTP-MTP charge term, but these values will ultimately agree if damping
    is disabled. Need extra flexibility for Z-MTP and MTP-MTP terms when
    damping.
    """
    if not match_cliff:
        c_qQ, c_uQ, c_QQ = (1.0 / 3.0), (1.0 / 3.0), (1.0 / 9.0)
    else:
        c_qQ, c_uQ, c_QQ = 1.0, 1.0, 1.0
    T0, T1, T2, T3, T4 = T_cart(RA, RB, alpha_i, alpha_j)
    E_ZA_MB = None
    E_ZB_MA = None
    E_ZA_ZB = None
    if ZA is not None and ZB is not None:
        qA -= ZA
        qB -= ZB

    # Most inputs will already be traceless, but we can ensure this is the case
    if not traceless:
        thetaA = 0.5 * (thetaA + np.swapaxes(thetaA, -1, -2))
        traceA = np.trace(thetaA)
        thetaA[0, 0] -= traceA / 3.0
        thetaA[1, 1] -= traceA / 3.0
        thetaA[2, 2] -= traceA / 3.0
        thetaB = 0.5 * (thetaB + np.swapaxes(thetaB, -1, -2))
        traceB = np.trace(thetaB)
        thetaB[0, 0] -= traceB / 3.0
        thetaB[1, 1] -= traceB / 3.0
        thetaB[2, 2] -= traceB / 3.0

    # AP2 code had factors of 1/3, -1/3, 1/9 in qQ, uQ, QQ terms; however,
    # these make the energies disagree with CLIFF. CLIFF achieves better
    # agreement with SAPT0 elst, so which is right?
    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * c_qQ  # * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(
        T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))
    ) * -1.0 * c_uQ # * (-1.0 / 3.0)

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * c_QQ # * (1.0 / 9.0)
    """
ZA-ZB
8 8 0.19095216 12.22093793
8 1 0.16253609 1.30028871
8 1 0.16256201 1.30049608
1 8 0.16554119 1.32432951
1 1 0.14040889 0.14040889
1 1 0.14042195 0.14042195
1 8 0.29231273 2.33850186
1 1 0.22426564 0.22426564
1 1 0.22430324 0.22430324
ZA-MB
  Z*q: -13.60338385, Z*mu: 0.04002783, Z*theta: -0.02108684
  Z*q: -0.71177466, Z*mu: -0.00179446, Z*theta: -0.00035156
  Z*q: -0.71189083, Z*mu: -0.00179104, Z*theta: -0.00035197
  Z*q: -1.47416487, Z*mu: 0.00500052, Z*theta: -0.00097779
  Z*q: -0.07686180, Z*mu: -0.00015570, Z*theta: -0.00000937
  Z*q: -0.07686924, Z*mu: -0.00015547, Z*theta: -0.00000938
  Z*q: -2.60075784, Z*mu: 0.01221299, Z*theta: -0.00873778
  Z*q: -0.12269015, Z*mu: -0.00055956, Z*theta: -0.00007041
  Z*q: -0.12271109, Z*mu: -0.00055906, Z*theta: -0.00007051
ZB-MA
  Z*q: -13.59981718, Z*mu: -0.03695603, Z*theta: 0.02303718
  Z*q: -0.72544235, Z*mu: -0.00596415, Z*theta: 0.00084076
  Z*q: -1.28021441, Z*mu: 0.01778713, Z*theta: 0.00286522
  Z*q: -1.44702582, Z*mu: -0.00251122, Z*theta: 0.00186877
  Z*q: -0.07691623, Z*mu: -0.00054367, Z*theta: 0.00005876
  Z*q: -0.12312702, Z*mu: 0.00136705, Z*theta: 0.00015513
  Z*q: -1.44725659, Z*mu: -0.00251069, Z*theta: 0.00186976
  Z*q: -0.07692339, Z*mu: -0.00054379, Z*theta: 0.00005877
  Z*q: -0.12314757, Z*mu: 0.00136752, Z*theta: 0.00015518
MTP-MTP
8-8: 15.130525 = 15.136812 + -0.003875 + -0.002412
8-1: 0.794777 = 0.792050 + 0.003380 + -0.000653
8-1: 0.794902 = 0.792179 + 0.003376 + -0.000653
1-8: 0.810941 = 0.807460 + 0.003860 + -0.000379
1-1: 0.042461 = 0.042103 + 0.000384 + -0.000027
1-1: 0.042465 = 0.042108 + 0.000384 + -0.000027
1-8: 1.393806 = 1.418173 + -0.025492 + 0.001125
1-1: 0.066756 = 0.067246 + -0.000442 + -0.000048
1-1: 0.066767 = 0.067257 + -0.000443 + -0.000048
Elst: 12056.938032 + -12224.225863 + -11858.340758 + 12012.663869 = -12.964720

MTP-MTP
5.236914, lam1=0.99986220, lam3=0.99849818
 lam5=0.99403030, lam7=0.98443204, lam9=0.96833249
8-8: 15.130525 = 15.136812 + -0.003875 + -0.002412
6.152480, lam1=0.99989728, lam3=0.99886701
 lam5=0.99543562, lam7=0.98792539, lam9=0.97507437
8-1: 0.794777 = 0.792050 + 0.003380 + -0.000653
6.151499, lam1=0.99989711, lam3=0.99886537
 lam5=0.99542963, lam7=0.98791104, lam9=0.97504752
8-1: 0.794902 = 0.792179 + 0.003376 + -0.000653
6.040793, lam1=0.99987677, lam3=0.99866414
 lam5=0.99470222, lam7=0.98618291, lam9=0.97183915
1-8: 0.810941 = 0.807460 + 0.003860 + -0.000379
7.122056, lam1=0.99994722, lam3=0.99937064
 lam5=0.99728388, lam7=0.99236768, lam9=0.98339250
1-1: 0.042461 = 0.042103 + 0.000384 + -0.000027
7.121394, lam1=0.99994717, lam3=0.99937006
 lam5=0.99728159, lam7=0.99236184, lam9=0.98338093
1-1: 0.042465 = 0.042108 + 0.000384 + -0.000027
3.420994, lam1=0.99169159, lam3=0.94694657
 lam5=0.86785321, lam7=0.76996490, lam9=0.66912774
1-8: 1.393806 = 1.418173 + -0.025492 + 0.001125
4.458998, lam1=0.99706199, lam3=0.97770563
 lam5=0.93580352, lam7=0.87399866, lam9=0.80005434
1-1: 0.066756 = 0.067246 + -0.000442 + -0.000048
4.458250, lam1=0.99705874, lam3=0.97768455
 lam5=0.93575170, lam7=0.87391187, lam9=0.79993664
1-1: 0.066767 = 0.067257 + -0.000443 + -0.000048
Elst: 12056.938032 + -12224.225863 + -11858.340758 + 12012.663869 = -12.964720

MTP-MTP
T2 = [ 1.37917502e-02  6.28902335e-04 -6.71242175e-05  6.28902335e-04
 -6.93311268e-03 -2.03503280e-06 -6.71242175e-05 -2.03503280e-06
 -6.95196217e-03]
T3 = [[-7.71632368e-03 -4.74231116e-04  5.06157965e-05]
 [-4.74231116e-04  3.94491880e-03  1.92331462e-06]
 [ 5.06157965e-05  1.92331462e-06  3.77140488e-03]
 [-4.74231116e-04  3.94491880e-03  1.92331462e-06]
 [ 3.94491880e-03  3.59891867e-04 -1.27651604e-05]
 [ 1.92331462e-06 -1.27651604e-05  1.14339249e-04]
 [ 5.06157965e-05  1.92331462e-06  3.77140488e-03]
 [ 1.92331462e-06 -1.27651604e-05  1.14339249e-04]
 [ 3.77140488e-03  1.14339249e-04 -3.78506361e-05]]
8-8: 15.130525 = 15.136812 + -0.003875 + -0.002412

q-Q =  -0.0021665167478815903
mu-Q 0.00012523309182835613
Q-Q -0.0003702415256548531
8-8: 15.130525 = 15.136812 + -0.003875 + -0.002412
MTP-MTP                                                                                                                                                                                        
    """

    # MTP-MTP
    # print(f"{ZA}-{ZB}: {E_qq: .6f} = {E_qq + E_qu + E_uu + E_qQ + E_uQ + E_QQ: .6f} + {E_qu: .6f} + {E_uu: .6f} + {E_qQ: .6f} + {E_uQ: .6f} + {E_QQ: .6f}")
    # print(f"q-Q =", E_qQ)
    # print(f"mu-Q =", E_uQ)
    # print(f"Q-Q =", E_QQ)


    if ZA is not None and ZB is not None:
        # Nuclear attraction terms
        T0, _, _ = T_cart_Z_MTP(RA, RB, None)
        E_ZA_ZB = T0 * ZA * ZB
        # print(f"{ZA} {ZB} {T0:.8f} {E_ZA_ZB: .8f}")

        # Only update to specific T's if damping
        if alpha_i is not None and alpha_j is not None:
            T0, T1, T2 = T_cart_Z_MTP(RA, RB, alpha_j)
        # A: Nuclear - charge, Nuclear - dipole, Nuclear - theta
        E_ZA_qB = T0 * ZA * qB
        # print(f"{ZA} {qB:.4f} {T0:.4f} {E_ZA_qB: .4f}")
        E_ZA_uB = np.sum(T1 * ZA * muB)
        E_ZA_QB = np.sum(T2 * ZA * thetaB * c_qQ)
        E_ZA_MB = E_ZA_qB + E_ZA_uB + E_ZA_QB
        # print(f"  Z*q: {E_ZA_qB: .8f}, Z*mu: {E_ZA_uB: .8f}, Z*theta: {E_ZA_QB: .8f}")
        if alpha_i is not None:
            T0, T1, T2 = T_cart_Z_MTP(RA, RB, alpha_i)
        # B: Nuclear - charge, Nuclear - dipole, Nuclear - theta
        E_ZB_qA = T0 * ZB * qA
        print(f"{ZB} {qA:.6f} {T0:.6f} {E_ZB_qA:.6f}")
        E_ZB_uA = np.sum(-T1 * ZB * muA)
        E_ZB_QA = np.sum(T2 * ZB * thetaA * c_qQ)
        E_ZB_MA = E_ZB_qA + E_ZB_uA + E_ZB_QA
        # print(f"  Z*q: {E_ZB_MA: .8f}, Z*mu: {np.sum(-T1 * ZB * muA): .8f}, Z*theta: {np.sum(T2 * ZB * thetaA * c_qQ): .8f}")

    return E_qq, E_qu, E_uu, E_qQ, E_uQ, E_QQ, E_ZA_ZB, E_ZA_MB, E_ZB_MA


def interaction_tensor(coord1, coord2, cell=None):
    """Return interaction tensor up to quadrupoles between two atom coordinates"""
    # Indices for MTP moments:
    # 00  01  02  03  04  05  06  07  08  09  10  11  12
    #  .,  x,  y,  z, xx, xy, xz, yx, yy, yz, zx, zy, zz
    if cell:
        vec = cell.pbc_distance(coord1, coord2)
    else:
        vec = coord2 - coord1
    r = np.linalg.norm(vec)
    r2 = r**2
    r4 = r2**2
    ri = 1.0 / r
    ri2 = ri**2
    ri3 = ri**3
    ri5 = ri**5
    ri7 = ri**7
    ri9 = ri**9
    x = vec[0]
    y = vec[1]
    z = vec[2]
    x2 = x**2
    y2 = y**2
    z2 = z**2
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2
    it = np.zeros((13, 13))
    # Charge charge
    it[0, 0] = ri
    # Charge dipole
    it[0, 1] = -x * ri3
    it[0, 2] = -y * ri3
    it[0, 3] = -z * ri3
    # Charge quadrupole
    it[0, 4] = (3 * x2 - r2) * ri5  # xx
    it[0, 5] = 3 * x * y * ri5  # xy
    it[0, 6] = 3 * x * z * ri5  # xz
    it[0, 7] = it[0, 5]  # yx
    it[0, 8] = (3 * y2 - r2) * ri5  # yy
    it[0, 9] = 3 * y * z * ri5  # yz
    it[0, 10] = it[0, 6]  # zx
    it[0, 11] = it[0, 9]  # zy
    it[0, 12] = -it[0, 4] - it[0, 8]  # zz
    # Dipole dipole
    it[1, 1] = -it[0, 4]  # xx
    it[1, 2] = -it[0, 5]  # xy
    it[1, 3] = -it[0, 6]  # xz
    it[2, 2] = -it[0, 8]  # yy
    it[2, 3] = -it[0, 9]  # yz
    it[3, 3] = -it[1, 1] - it[2, 2]  # zz
    # Dipole quadrupole
    it[1, 4] = -3 * x * (3 * r2 - 5 * x2) * ri7  # xxx
    it[1, 5] = it[1, 7] = it[2, 4] = -3 * y * (r2 - 5 * x2) * ri7  # xxy xyx yxx
    it[1, 6] = it[1, 10] = it[3, 4] = -3 * z * (r2 - 5 * x2) * ri7  # xxz xzx zxx
    it[1, 8] = it[2, 5] = it[2, 7] = -3 * x * (r2 - 5 * y2) * ri7  # xyy yxy yyx
    it[1, 9] = it[1, 11] = it[2, 6] = it[2, 10] = it[3, 5] = it[3, 7] = (
        15 * x * y * z * ri7
    )  # xyz xzy yxz yzx zxy zyx
    it[1, 12] = it[3, 6] = it[3, 10] = -it[1, 4] - it[1, 8]  # xzz zxz zzx
    it[2, 8] = -3 * y * (3 * r2 - 5 * y2) * ri7  # yyy
    it[2, 9] = it[2, 11] = it[3, 8] = -3 * z * (r2 - 5 * y2) * ri7  # yyz yzy zyy
    it[2, 12] = it[3, 9] = it[3, 11] = -it[1, 5] - it[2, 8]  # yzz zyz zzy
    it[3, 12] = -it[1, 6] - it[2, 9]  # zzz
    # Quadrupole quadrupole
    it[4, 4] = (105 * x4 - 90 * x2 * r2 + 9 * r4) * ri9  # xxxx
    it[4, 5] = it[4, 7] = 15 * x * y * (7 * x2 - 3 * r2) * ri9  # xxxy xxyx
    it[4, 6] = it[4, 10] = 15 * x * z * (7 * x2 - 3 * r2) * ri9  # xxxz xxzx
    it[4, 8] = it[5, 5] = it[5, 7] = it[7, 7] = (
        105 * x2 * y2 + 15 * z2 * r2 - 12 * r4
    ) * ri9  # xxyy xyxy xyyx yxyx
    it[4, 9] = it[4, 11] = it[5, 6] = it[5, 10] = it[6, 7] = it[7, 10] = (
        15 * y * z * (7 * x2 - 1 * r2) * ri9
    )  # xxyz xxzy xyxz xyzx xzyx yxzx
    it[4, 12] = it[6, 6] = it[6, 10] = it[10, 10] = (
        -it[4, 4] - it[4, 8]
    )  # xxzz xzxz xzzx zxzx
    it[5, 8] = it[7, 8] = 15 * x * y * (7 * y2 - 3 * r2) * ri9  # xyyy yxyy
    it[5, 9] = it[5, 11] = it[6, 8] = it[7, 9] = it[7, 11] = it[8, 10] = (
        15 * x * z * (7 * y2 - r2) * ri9
    )  # xyyz xyzy xzyy yxyz yxzy yyzx
    it[5, 12] = it[6, 9] = it[6, 11] = it[7, 12] = it[9, 10] = it[10, 11] = (
        -it[4, 5] - it[5, 8]
    )  # xyzz xzyz xzzy yxzz yzzx zxzy
    it[6, 12] = it[10, 12] = -it[4, 6] - it[5, 9]  # xzzz zxzz
    it[8, 8] = (105 * y4 - 90 * y2 * r2 + 9 * r4) * ri9  # yyyy
    it[8, 9] = it[8, 11] = 15 * y * z * (7 * y2 - 3 * r2) * ri9  # yyyz yyzy
    it[8, 12] = it[9, 9] = it[9, 11] = it[11, 11] = (
        -it[4, 8] - it[8, 8]
    )  # yyzz yzyz yzzy zyzy
    it[9, 12] = it[11, 12] = -it[4, 9] - it[8, 9]  # yzzz zyzz
    it[12, 12] = -it[4, 12] - it[8, 12]  # zzzz
    # Symmetrize
    it = it + it.T - np.diag(it.diagonal())
    # Some coefficients need to be multiplied by -1
    for i in range(1, 4):
        for j in range(0, 1):
            it[i, j] *= -1.0
    for i in range(4, 13):
        for j in range(1, 4):
            it[i, j] *= -1.0
    return it


def eval_interaction(RA, qA, muA, thetaA, RB, qB, muB, thetaB, traceless=False):
    T0, T1, T2, T3, T4 = T_cart(RA, RB)

    # Most inputs will already be traceless, but we can ensure this is the case
    if not traceless:
        traceA = np.trace(thetaA)
        thetaA[0, 0] -= traceA / 3.0
        thetaA[1, 1] -= traceA / 3.0
        thetaA[2, 2] -= traceA / 3.0
        traceB = np.trace(thetaB)
        thetaB[0, 0] -= traceB / 3.0
        thetaB[1, 1] -= traceB / 3.0
        thetaB[2, 2] -= traceB / 3.0

    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(
        T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))
    ) * (-1.0 / 3.0)

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * (1.0 / 9.0)

    # partial-charge electrostatic energy
    E_q = E_qq

    # dipole correction
    E_u = E_qu + E_uu

    # quadrupole correction
    E_Q = E_qQ + E_uQ + E_QQ

    return E_q + E_u + E_Q


def eval_dimer2(RA, RB, ZA, ZB, QA, QB):
    maskA = ZA >= 1
    maskB = ZB >= 1

    # Keep R in a.u. (molden convention)
    RA_temp = RA[maskA] * 1.88973
    RB_temp = RB[maskB] * 1.88973
    ZA_temp = ZA[maskA]
    ZB_temp = ZB[maskB]
    QA_temp = QA[maskA]
    QB_temp = QB[maskB]

    quadrupole_A = []
    quadrupole_B = []

    for ia in range(len(RA_temp)):
        # QA_temp[ia][4:10] = (3.0/2.0) * qpole_redundant(QA_temp[ia][4:10])
        quadrupole_A.append((3.0 / 2.0) * qpole_redundant(QA_temp[ia][4:10]))

    for ib in range(len(RB_temp)):
        # QB_temp[ib][4:10] = (3.0/2.0) * qpole_redundant(QB_temp[ib][4:10])
        quadrupole_B.append((3.0 / 2.0) * qpole_redundant(QB_temp[ib][4:10]))

    total_energy = 0.0

    # calculate multipole electrostatics for each atom pair
    for ia in range(len(RA_temp)):
        for ib in range(len(RB_temp)):
            rA = RA_temp[ia]
            qA = QA_temp[ia]

            rB = RB_temp[ib]
            qB = QB_temp[ib]

            pair_energy = eval_interaction(
                rA,
                qA[0],
                qA[1:4],
                # qA[4:10],
                quadrupole_A[ia],
                rB,
                qB[0],
                qB[1:4],
                # qB[4:10])
                quadrupole_B[ib],
            )

            total_energy += pair_energy

    Har2Kcalmol = 627.5094737775374055927342256

    return total_energy * Har2Kcalmol


def eval_dimer(RA, RB, ZA, ZB, QA, QB):
    # Keep R in a.u. (molden convention)
    RA_temp = RA * 1.88973
    RB_temp = RB * 1.88973

    total_energy = 0.0

    maskA = ZA >= 1
    maskB = ZB >= 1

    pair_mat = np.zeros((int(np.sum(maskA, axis=0)), int(np.sum(maskB, axis=0))))

    # calculate multipole electrostatics for each atom pair
    for ia in range(len(RA_temp)):
        for ib in range(len(RB_temp)):
            rA = RA_temp[ia]
            zA = ZA[ia]
            qA = QA[ia]

            rB = RB_temp[ib]
            zB = ZB[ib]
            qB = QB[ib]

            if (zA == 0) or (zB == 0):
                continue

            pair_energy = eval_interaction(
                rA,
                qA[0],
                qA[1:4],
                (3.0 / 2.0) * qpole_redundant(qA[4:10]),
                rB,
                qB[0],
                qB[1:4],
                (3.0 / 2.0) * qpole_redundant(qB[4:10]),
            )
            total_energy += pair_energy
            pair_mat[ia][ib] = pair_energy

    Har2Kcalmol = 627.5094737775374055927342256

    return total_energy * Har2Kcalmol, pair_mat * Har2Kcalmol


libmbd_vwd_params = pd.read_csv(
    # osp.join(current_file_path, "data", "vdw-params.csv"),
    resources.files(
        "apnet_pt",
    ).joinpath("data", "vdw-params.csv"),
    header=0,
    index_col=0,
    sep=",",
    nrows=102,
)
free_atom_polarizabilities = {
    # el: v for el, v in zip(libmbd_vwd_params['Z'], libmbd_vwd_params['alpha_0(TS)'])
    el: v
    for el, v in zip(libmbd_vwd_params["Z"], libmbd_vwd_params["alpha_0(BG)"])
}


def dimer_induced_dipole(
    qcel_dimer: qcel.models.Molecule,
    qA: np.ndarray,
    muA: np.ndarray,
    thetaA: np.ndarray,
    qB: np.ndarray,
    muB: np.ndarray,
    thetaB: np.ndarray,
    hirshfeld_volume_ratio_A: np.ndarray,
    hirshfeld_volume_ratio_B: np.ndarray,
    valence_widths_A: np.ndarray,
    valence_widths_B: np.ndarray,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    # CLIFF omega parameter = 0.7 for fewer iterations
    omega: float = 0.7,
    # CLIFF Thole damping parameter = 0.39
    thole_damping_param: float = 0.39,
) -> float:
    """
    Calculate the induced dipole interaction energy between two molecules using
    their multipole moments and Hirshfeld volume ratios. Follow classical
    induction model from this paper:
    https://pubs.aip.org/aip/jcp/article/154/18/184110/200216/CLIFF-A-component-based-machine-learned
    """

    # Get molecular fragments
    molA = qcel_dimer.get_fragment(0)
    molB = qcel_dimer.get_fragment(1)
    alpha_0_A = np.array([free_atom_polarizabilities[i] for i in molA.atomic_numbers])
    alpha_0_B = np.array([free_atom_polarizabilities[i] for i in molB.atomic_numbers])
    hirshfeld_volume_ratio_A = hirshfeld_volume_ratio_A.flatten()
    hirshfeld_volume_ratio_B = hirshfeld_volume_ratio_B.flatten()

    # Get atomic positions (in bohr)
    RA = molA.geometry
    RB = molB.geometry
    np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

    # Calculate atomic polarizabilities using Hirshfeld volume ratios
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B

    # Combine all atoms and properties
    R_all = np.vstack([RA, RB])
    alpha_all = np.vstack([alpha_A.reshape(-1, 1), alpha_B.reshape(-1, 1)]).flatten()
    q_all = np.concatenate([qA, qB]).flatten()
    mu_all = np.concatenate([muA, muB])
    theta_all = np.concatenate([thetaA, thetaB])

    n_atoms_A = len(RA)
    n_atoms_B = len(RB)
    n_atoms_total = n_atoms_A + n_atoms_B

    # Interaction tensor between M_i*T_ij*M_j
    T_abij = np.zeros((n_atoms_total, n_atoms_total, 13, 13))
    M = np.zeros((n_atoms_total, 13))
    M[:, 0] = q_all  # Charge
    M[:, 1:4] = mu_all  # Dipole
    M[:, 4:13] = theta_all.reshape(n_atoms_total, 9)  # Quadrupole (flattened)

    # Multipoles on molecule A
    M_A = M[:n_atoms_A, :]
    # Multipoles on molecule B
    M_B = M[n_atoms_A:, :]

    # Initialize interaction tensors
    for i in range(n_atoms_total):
        for j in range(n_atoms_total):
            if i == j:
                T_abij[i, j, :, :] = np.zeros((13, 13))
                continue
            T0, T1, T2, T3, T4 = T_cart_Thole_damping(
                R_all[i], R_all[j], alpha_all[i], alpha_all[j], thole_damping_param
            )
            # Added constants to agree with eval_interaction terms
            # T_abij[i, j, 0, 0] = T0
            # T_abij[i, j, 1:4, 0] = -1.0 * T1
            # T_abij[i, j, 0, 1:4] = T1.T
            # T_abij[i, j, 1:4, 1:4] = T2
            # T_abij[i, j, 1:4, 4:13] = -1 / 3 * T3.reshape(3, 9)
            # T_abij[i, j, 4:13, 1:4] = 1 / 3 * T3.T.reshape(9, 3)
            # T_abij[i, j, 4:13, 4:13] = 1.0 / 9.0 * T4.reshape(9, 9)
            # T_abij[i, j, 0, 4:13] = 1 / 3 * T2.reshape(9)
            # T_abij[i, j, 4:13, 0] = 1 / 3 * T2.reshape(9)

            T_abij[i, j, 0, 0] = T0
            T_abij[i, j, 1:4, 0] = T1
            T_abij[i, j, 0, 1:4] = T1
            T_abij[i, j, 1:4, 1:4] = T2
            T_abij[i, j, 1:4, 4:13] = T3.reshape(3, 9)
            T_abij[i, j, 4:13, 1:4] = T3.T.reshape(9, 3)
            T_abij[i, j, 4:13, 4:13] = T4.reshape(9, 9)
            T_abij[i, j, 0, 4:13] =  T2.reshape(9)
            T_abij[i, j, 4:13, 0] =  T2.reshape(9)

    E_qq = float(
        (
            np.einsum(
                "ai,abij,bj->",
                M_A[:, 0:1],
                T_abij[:n_atoms_A, n_atoms_A:, 0:1, 0:1],
                M_B[:, 0:1],
            )
        )
        * constants.h2kcalmol
    )
    print(f"{E_qq=:.6f}")
    E_qu = float(
        (
            np.einsum(
                "a,abj,bj->",
                M_A[:, 0],
                T_abij[:n_atoms_A, n_atoms_A:, 0, 1:4],
                M_B[:, 1:4],
            )
            - np.einsum(
                "ai,abi,b->",
                M_A[:, 1:4],
                T_abij[:n_atoms_A, n_atoms_A:, 1:4, 0],
                M_B[:, 0],
            )
        )
        * constants.h2kcalmol
    )
    print(f"{E_qu=:.6f}")
    E_uu = float(
        -1.0 * (
            np.einsum(
                "ai,abij,bj->",
                M_A[:, 1:4],
                T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
                M_B[:, 1:4],
            )
        )
        * constants.h2kcalmol
    )
    print(f"{E_uu=:.6f}")
    E_qQ = float(
        1/3*(
            np.einsum(
                "ai,abij,bj->",
                M_A[:, 0:1],
                T_abij[:n_atoms_A, n_atoms_A:, 0:1, 4:],
                M_B[:, 4:],
            )
            + np.einsum(
                "ai,abij,bj->",
                M_A[:, 4:],
                T_abij[:n_atoms_A, n_atoms_A:, 4:, 0:1],
                M_B[:, 0:1],
            )
        )
        * constants.h2kcalmol
    )
    print(f"{E_qQ=:.6f}")
    E_uQ = float(
        1/3*(
            np.einsum(
                "ai,abij,bj->",
                M_A[:, 1:4],
                T_abij[:n_atoms_A, n_atoms_A:, 1:4, 4:],
                M_B[:, 4:],
            )
            + np.einsum(
                "ai,abij,bj->",
                M_A[:, 4:],
                T_abij[:n_atoms_A, n_atoms_A:, 4:, 1:4],
                M_B[:, 1:4],
            )
        )
        * constants.h2kcalmol
    )
    print(f"{E_uQ=:.6f}")
    mu_induced_0 = np.zeros((n_atoms_total, 3))
    mu_induced_0_A = mu_induced_0[:n_atoms_A, :]
    mu_induced_0_B = mu_induced_0[n_atoms_A:, :]
    mu_induced_0_A[:, :] = np.einsum(
        "a,abij,bj->ai", alpha_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B
    )
    mu_induced_0_B[:, :] = np.einsum(
        "b,baij,ai->bj", alpha_B, T_abij[n_atoms_A:, :n_atoms_A, :, 1:4], M_A
    )
    # Self-consistent induced dipole iterations
    mu_induced = mu_induced_0.copy()
    M_induced_0 = M.copy()
    M_induced_0[:n_atoms_A, 1:4] = mu_induced_0_A
    M_induced_0[n_atoms_A:, 1:4] = mu_induced_0_B
    M_A_induced_0 = M_induced_0[:n_atoms_A, :]
    M_B_induced_0 = M_induced_0[n_atoms_A:, :]
    M_induced = M.copy()
    for iteration in range(max_iterations):
        mu_induced_old = mu_induced.copy()
        mu_sum = np.zeros_like(mu_induced)
        for i in range(n_atoms_total):
            mu_sum[i] = alpha_all[i] * np.einsum(
                "nij,nj->i",
                T_abij[i, :, 1:4, 1:4],
                M_induced[:, 1:4],
            )
        # print(f"{mu_sum[0, :] = }")
        mu_sum += mu_induced_0
        mu_induced = (1 - omega) * mu_induced_old + omega * (mu_sum)
        M_induced[:, 1:4] = mu_induced
        # print(f"{mu_induced[0, :] = }")
        # break
        # Check convergence
        delta = np.linalg.norm(mu_induced - mu_induced_old)
        if delta < convergence_threshold:
            print(f"   Converged after {iteration + 1} iterations.")
            break
    mu_induced_A = mu_induced[:n_atoms_A, :]
    mu_induced_B = mu_induced[n_atoms_A:, :]
    print("mu:")
    print(mu_all)
    print("mu(0):")
    print(mu_induced_0)
    print("mu(n):")
    print(mu_induced)
    # diff
    print(f"Change in induced dipoles over {iteration = }:")
    print(mu_induced_0 - mu_induced_old)

    # Calculate induction energy
    E_ind = 0.0
    E_0_ind = (
        float(
            np.einsum(
                "abji,bi,aj->",
                T_abij[:n_atoms_A, n_atoms_A:, :, 1:4],
                mu_induced_0_B,
                M_A_induced_0,
            )
            - np.einsum(
                "abij,ai,bj->",
                T_abij[:n_atoms_A, n_atoms_A:, 1:4, :],
                mu_induced_0_A,
                M_B_induced_0,
            )
        )
        * constants.h2kcalmol
    )
    print(f"{E_0_ind = }")
    E_ind_BA = np.einsum(
            "bi,baij,aj->", mu_induced_B, T_abij[n_atoms_A:, :n_atoms_A, 1:4, :], M_A
    ) * constants.h2kcalmol
    E_ind_AB = np.einsum(
            "ai,baij,bj->", mu_induced_A, T_abij[n_atoms_A:, :n_atoms_A, 1:4, :], M_B
    ) * constants.h2kcalmol
    print(f" {E_ind_AB = :.6f}\n {E_ind_BA = :.6f}")
    E_ind = float(
        E_ind_AB - E_ind_BA
    )
    # E_ind *= constants.h2kcalmol  # * 0.5
    return E_ind

if __name__ == "__main__":
    T_cart()
