"""
Functions for evaluating electrostatics between atom-centered multipoles
"""

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


def T_cart(RA, RB):
    dR = RB - RA
    R = np.linalg.norm(dR)

    delta = np.identity(3)

    T0 = R**-1
    T1 = (R**-3) * (-1.0 * dR)
    T2 = (R**-5) * (3 * np.outer(dR, dR) - R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    T3 = (
        (R**-7)
        * -1.0
        * (
            15 * np.multiply.outer(np.outer(dR, dR), dR)
            - 3 * R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
        )
    )

    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    T4 = (R**-9) * (
        105 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - 15
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
        + 3 * (R**4) * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
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


def eval_qcel_dimer_individual(mol_dimer, qA, muA, thetaA, qB, muB, thetaB) -> float:
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
    mol_dimer, qA, muA, thetaA, qB, muB, thetaB
) -> Tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.
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

            E_qq, E_qu, E_uu, E_qQ, E_uQ, E_QQ = eval_interaction_individual_components(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j
            )
            E_qqs[i, j] = E_qq
            E_qus[i, j] = E_qu
            E_uus[i, j] = E_uu
            E_qQs[i, j] = E_qQ
            E_uQs[i, j] = E_uQ
            E_QQs[i, j] = E_QQ
    total_energy = (
        np.sum(E_qqs)
        + np.sum(E_qus)
        + np.sum(E_uus)
        + np.sum(E_qQs)
        + np.sum(E_uQs)
        + np.sum(E_QQs)
    )
    total_energy *= constants.h2kcalmol
    E_qqs *= constants.h2kcalmol
    E_qus *= constants.h2kcalmol
    E_uus *= constants.h2kcalmol
    E_qQs *= constants.h2kcalmol
    E_uQs *= constants.h2kcalmol
    E_QQs *= constants.h2kcalmol
    return total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs


def eval_interaction_individual(
    RA, qA, muA, thetaA, RB, qB, muB, thetaB, traceless=False
):
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
    return E_q, E_u, E_Q


def eval_interaction_individual_components(
    RA, qA, muA, thetaA, RB, qB, muB, thetaB, traceless=False
):
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
    return E_qq, E_qu, E_uu, E_qQ, E_uQ, E_QQ


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
    ri = 1./r
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
    it = np.zeros((13,13))
    # Charge charge
    it[0,0] = ri
    # Charge dipole
    it[0,1] = -x*ri3
    it[0,2] = -y*ri3
    it[0,3] = -z*ri3
    # Charge quadrupole
    it[0,4] = (3*x2-r2)*ri5 # xx
    it[0,5] = 3*x*y*ri5 # xy
    it[0,6] = 3*x*z*ri5 # xz
    it[0,7] = it[0,5] # yx
    it[0,8] = (3*y2-r2)*ri5 # yy
    it[0,9] = 3*y*z*ri5 # yz
    it[0,10] = it[0,6] # zx
    it[0,11] = it[0,9] # zy
    it[0,12] = -it[0,4] -it[0,8] # zz
    # Dipole dipole
    it[1,1] = -it[0,4] # xx
    it[1,2] = -it[0,5] # xy
    it[1,3] = -it[0,6] # xz
    it[2,2] = -it[0,8] # yy
    it[2,3] = -it[0,9] # yz
    it[3,3] = -it[1,1] -it[2,2] # zz
    # Dipole quadrupole
    it[1,4] = -3*x*(3*r2-5*x2)*ri7 # xxx
    it[1,5] = it[1, 7] = it[2,4] = -3*y*(r2-5*x2)*ri7 # xxy xyx yxx
    it[1,6] = it[1,10] = it[3,4] = -3*z*(r2-5*x2)*ri7 # xxz xzx zxx
    it[1,8] = it[2,5] = it[2,7] = -3*x*(r2-5*y2)*ri7 # xyy yxy yyx
    it[1,9] = it[1,11] = it[2,6] = it[2,10] = it[3,5] = \
        it[3,7] = 15*x*y*z*ri7 # xyz xzy yxz yzx zxy zyx
    it[1,12] = it[3,6] = it[3,10] = -it[1,4] -it[1,8] # xzz zxz zzx
    it[2,8] = -3*y*(3*r2-5*y2)*ri7 # yyy
    it[2,9] = it[2,11] = it[3,8] = -3*z*(r2-5*y2)*ri7 # yyz yzy zyy
    it[2,12] = it[3,9] = it[3,11] = -it[1,5] -it[2,8] # yzz zyz zzy
    it[3,12] = -it[1,6] -it[2,9] # zzz
    # Quadrupole quadrupole
    it[4,4] = (105*x4-90*x2*r2+9*r4)*ri9 # xxxx
    it[4,5] = it[4,7] =  15*x*y*(7*x2-3*r2)*ri9 # xxxy xxyx
    it[4,6] = it[4,10] = 15*x*z*(7*x2-3*r2)*ri9 # xxxz xxzx
    it[4,8] = it[5,5] = it[5,7] = it[7,7] = \
        (105*x2*y2+15*z2*r2-12*r4)*ri9 # xxyy xyxy xyyx yxyx
    it[4,9] = it[4,11] = it[5,6] = it[5,10] = it[6,7] = it[7,10] = \
        15*y*z*(7*x2-1*r2)*ri9 # xxyz xxzy xyxz xyzx xzyx yxzx
    it[4,12] = it[6,6] = it[6,10] = it[10,10] = \
        -it[4,4] -it[4,8] # xxzz xzxz xzzx zxzx
    it[5,8] = it[7,8] = 15*x*y*(7*y2-3*r2)*ri9 # xyyy yxyy
    it[5,9] = it[5,11] = it[6,8] = it[7,9] = it[7,11] = it[8,10] = \
        15*x*z*(7*y2-r2)*ri9 # xyyz xyzy xzyy yxyz yxzy yyzx
    it[5,12] = it[6,9] = it[6,11] = it[7,12] = it[9,10] = it[10,11] = \
        -it[4,5] -it[5,8] # xyzz xzyz xzzy yxzz yzzx zxzy
    it[6,12] = it[10,12] = -it[4,6] -it[5,9] # xzzz zxzz
    it[8,8] = (105*y4-90*y2*r2+9*r4)*ri9 # yyyy
    it[8,9] = it[8,11] = 15*y*z*(7*y2-3*r2)*ri9 # yyyz yyzy
    it[8,12] = it[9,9] = it[9,11] = it[11,11] = \
        -it[4,8] -it[8,8] # yyzz yzyz yzzy zyzy
    it[9,12] = it[11,12] = -it[4,9] -it[8,9] # yzzz zyzz
    it[12,12] = -it[4,12] -it[8,12] # zzzz
    # Symmetrize
    it = it + it.T - np.diag(it.diagonal())
    # Some coefficients need to be multiplied by -1
    for i in range(1,4):
        for j in range(0,1):
            it[i,j] *= -1.
    for i in range(4,13):
        for j in range(1,4):
            it[i,j] *= -1.
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
    # print()
    # print(RA.shape, ZA.shape, QA.shape)
    # print(RB.shape, ZB.shape, QB.shape)

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


import pandas as pd
from importlib import resources

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

    # Get atomic positions (in bohr)
    RA = molA.geometry
    RB = molB.geometry
    Z = qcel_dimer.atomic_numbers
    ZA = molA.atomic_numbers
    ZB = molB.atomic_numbers


    # Calculate atomic polarizabilities using Hirshfeld volume ratios
    print(f"{hirshfeld_volume_ratio_A=}, {hirshfeld_volume_ratio_B=}")
    print(f"{alpha_0_A=}, {alpha_0_B=}")
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B

    # Combine all atoms and properties
    R_all = np.vstack([RA, RB])
    alpha_all = np.concatenate([alpha_A, alpha_B])
    print(f"{alpha_all=}")
    q_all = np.concatenate([qA, qB])
    # print(f"{muA=}, {muB=}")
    mu_all = np.concatenate([muA, muB])
    print(f"{mu_all=}")
    theta_all = np.concatenate([thetaA, thetaB])

    n_atoms_A = len(RA)
    n_atoms_B = len(RB)
    n_atoms_total = n_atoms_A + n_atoms_B

    # Compute all pairwise distances and interaction tensors
    distances = np.zeros((n_atoms_total, n_atoms_total))

    # Interaction tensor between M_i*T_ij*M_j
    T_abij = np.zeros((n_atoms_total, n_atoms_total, 13, 13))
    T_undamped = np.zeros((n_atoms_total, n_atoms_total, 13, 13))

    M = np.zeros((n_atoms_total, 13))
    M[:, 0] = q_all  # Charge
    # print(M[:, 0])
    # print(Z)
    # M[:, 0] += Z
    # print(M[:, 0])
    M[:, 1:4] = mu_all  # Dipole
    M[:, 4:13] = theta_all.reshape(n_atoms_total, 9)  # Quadrupole (flattened)

    # Multipoles on molecule A
    M_A = M[:n_atoms_A, :]
    # Multipoles on molecule B
    M_B = M[n_atoms_A:, :]

    # Apply Thole damping to interaction tensors
    def thole_damping(r_ij, alpha_i, alpha_j, a):
        """Apply Thole damping to interaction tensor"""
        # Compute damping factor
        u = r_ij / ((alpha_i * alpha_j) ** (1.0 / 6.0))
        damping_exp = -a * (u**3)
        # print(f"{u=:.2f}, {alpha_i=:.2f}, {alpha_j=:.2f}, {a=:.2f}, {damping_exp=:.2f}")
        return damping_exp

    # Initialize interaction tensors
    for i in range(n_atoms_total):
        for j in range(n_atoms_total):
            if i == j:
                T_abij[i, j, :, :] = np.zeros((13, 13))
                continue
            distances[i, j] = np.linalg.norm(R_all[i] - R_all[j])

            thole_damping_exp = thole_damping(
                distances[i, j], alpha_all[i], alpha_all[j], thole_damping_param
            )
            T0, T1, T2, T3, T4 = T_cart(R_all[i], R_all[j])
            # print(f"{T0.shape=}, {T1.shape=}, {T2.shape=}, {T3.shape=}, {T4.shape=}")
            # if i == 0 and j == 1:
            scale = (3 * thole_damping_param) / (4 * np.pi)
            f_damp = (1 - (np.exp(thole_damping_exp)))
            # print(f"{i}, {j}: {int(Z[i])}, {int(Z[j])}, {scale:.2f}, {f_damp=:.2f}, {T0=:.2f}, {distances[i, j]=:.2f}")
            # print(f"{T0.shape=}, {T1.shape=}, {T2.shape=}")
            # print(f"{T[i, j].shape=}")
            # print(f"{f_damp=}")

            # [N_atoms, N_atoms, dipole(3), multipole(13)], I don't think dipole dimension is getting set correctly...
            exp = thole_damping_exp
            l3 = (1 - np.exp(exp))
            l5 = (1 - (1 - exp) * np.exp(exp))
            l7 = (1 - (1.0 - exp + 0.6 * exp * exp) * np.exp(exp))

            # print(f"{l3=:.2f}, {l5=:.2f}, {l7=:.2f}")

            T_abij[i, j, 0, 0] = l3 * T0
            T_abij[i, j, 1:4, 0] = l3 * T1
            T_abij[i, j, 0, 1:4] = l3 * T1.T
            T_abij[i, j, 1:4, 1:4] = l5 * T2 * -1.0
            for d in range(1, 4):
                T_abij[i, j, d, 1] -= l3 * np.power(distances[i, j], -3.0)

            T_pair = interaction_tensor(
                R_all[i], R_all[j]
            )
            T_undamped[i, j, :, :] = T_pair.copy()

            # T_abij[i, j, :, :] = T_pair
            # T_abij[i, j, 0, 0] = l3 * T_abij[i, j, 0, 0]
            # T_abij[i, j, 1:4, 0] = l3 * T_abij[i, j, 1:4, 0]
            # T_abij[i, j, 0, 1:4] = l3 * T_abij[i, j, 0, 1:4]
            # T_abij[i, j, 1:4, 1:4] = l5 * T_abij[i, j, 1:4, 1:4]
            # for d in range(1, 4):
            #     T_abij[i, j, d, 1] -= l3 * np.power(distances[i, j], -3.0)
            # T_abij[i, j, 1:4, 4:13] = l5 * T_abij[i, j, 1:4, 4:13]
            # T_abij[i, j, 4:13, 1:4] = l5 * T_abij[i, j, 4:13, 1:4]
            # T_abij[i, j, 4:13, 4:13] *= l7

    # Not full elst term, but multipole elst. Need to include Z_iZ_j and
    # Z_iM_j, Z_jM_i terms, but agrees well enough with SAPT0 Elst to say T_ij
    # is seemingly correct
    E_elst = np.einsum(
        "ai,abij,bj->", M_A, T_undamped[n_atoms_A:, :n_atoms_A, :, :], M_B
    ) * constants.h2kcalmol
    E_elst += np.einsum(
        "ai,abij,bj->", M_B, T_undamped[:n_atoms_A, n_atoms_A:, :, :], M_A
    ) * constants.h2kcalmol

    # Temporarily limiting to only charge and dipole interactions
    # Also, lets examine the 2x2 electrostaics OH, OH interaction only to keep
    # it simple under *_q_mu vars
    M = M[:, :4]  # Keep only charge and dipole moments
    # print(f"{M = }")
    M_A = M[:n_atoms_A, :4]
    M_B = M[n_atoms_A:, :4]
    M_A_q_mu = M[:n_atoms_A, :]
    M_B_q_mu = M[n_atoms_A:, :]
    T_abij_q_mu = T_abij[:n_atoms_A, n_atoms_A:, :4, :4]
    T_abij = T_abij[:, :, :4, :4]
    # print(T_abij[0, 0, :, :])
    # print("q:\n",
    #     # np.einsum(
    #     #   "ai,abij,bj->ab", M_A_q_mu[:, 0:1], T_abij_q_mu[:, :, 0:1, 0:1], M_B_q_mu[:, 0:1]
    #     # )
    #     np.einsum(
    #       "abij,ai,bj->ab", T_abij_q_mu[:, :, 0:1, 0:1], M_A_q_mu[:, 0:1], M_B_q_mu[:, 0:1]
    #     )
    # )
    # print("q (A self):\n",
    #     np.einsum(
    #       "ai,abij,bj->ab", M_A[:, 0:1], T_abij[:n_atoms_A, :n_atoms_A, 0:1, 0:1], M_A[:, 0:1]
    #     )
    # )
    # print("q (B self):\n",
    #     np.einsum(
    #       "ai,abij,bj->ab", M_B[:, 0:1], T_abij[n_atoms_A:, n_atoms_A:, 0:1, 0:1], M_B[:, 0:1]
    #     )
    # )
    # print("mu:\n",
    #     np.einsum(
    #       "ai,abij,bj->ab", M_A_q_mu[:, 1:4], T_abij_q_mu[:, :, 1:4, 1:4], M_B_q_mu[:, 1:4]
    #     )
    # )
    # print("q_mu_A:\n",
    #     np.einsum(
    #       "ai,abij,bj->ab", M_A_q_mu[:, 0:1], T_abij_q_mu[:, :, 0:1, 1:4], M_B_q_mu[:, 1:4]
    #     )
    # )
    # print("q_mu_B:\n",
    #     np.einsum(
    #       "ai,abij,bj->ab", M_A_q_mu[:, 1:4], T_abij_q_mu[:, :, 1:4, 0:1], M_B_q_mu[:, 0:1]
    #     )
    # )
    # print(f"mu (A self):\n",
    #     np.einsum(
    #         "ai,abij,bj->ab", M_A[:, 1:4], T_abij[:n_atoms_A, :n_atoms_A, 1:4, 1:4], M_A[:, 1:4]
    #       )
    # )
    T_AB = T_abij[:n_atoms_A, n_atoms_A:, 1:4, :]
    T_BA = T_abij[n_atoms_A:, :n_atoms_A, :, 1:4]
    q_energies = float((
        np.einsum(
            "ai,abij,bj->", M_A[:, 0:1], T_abij[:n_atoms_A, n_atoms_A:, 0:1, :], M_B[:, 0:1]
        ) + np.einsum(
            "ai,abij,bj->", M_A[:, 0:1], T_abij[:n_atoms_A, n_atoms_A:, :, 0:1], M_B[:, 1:4]
        )
    ) * constants.h2kcalmol)
    print(f"{q_energies = }")
    mu_energies = float((
            # np.einsum(
            # "ai,abij,bj->", M_A[:, 1:4], T_AB, M_B
            # ) + np.einsum(
            #     "ai,abij,bj->", M_A, T_BA, M_B[:, 1:4]
            # )
            np.einsum(
            "ai,abij,bj->", M_A[:, 1:4], T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B
            ) + np.einsum(
                "ai,abij,bj->", M_A, T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], M_B[:, 1:4]
            )
    ) * constants.h2kcalmol)
    print(f"{mu_energies = }")
    mu_induced_0 = np.zeros((n_atoms_total, 3))
    mu_induced_0_A = mu_induced_0[:n_atoms_A, :]
    mu_induced_0_B = mu_induced_0[n_atoms_A:, :]

    # T_AB = T_abij[:n_atoms_A, n_atoms_A:, 1:4, :]
    # T_BA = T_abij[n_atoms_A:, :n_atoms_A, 1:4, :]
    mu_induced_0_A[:, :] = np.einsum(
        # "a,abij,bj->ai", alpha_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B,
        "abij,a,bj->ai", T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], alpha_A, M_B,
    )
    mu_induced_0_B[:, :]  = np.einsum(
        # "b,abji,aj->bi", alpha_B, T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], M_A
        "abji,b,aj->bi", T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], alpha_B, M_A
    )
    # Self-consistent field iteration
    mu_induced = mu_induced_0.copy()
    print(f"{mu_induced = }")
    M_induced = M.copy()
    M_A_induced = M_induced[:n_atoms_A, :4]
    M_B_induced = M_induced[n_atoms_A:, :4]
    for iteration in range(max_iterations):
        mu_induced_old = mu_induced.copy()
        # mu_sum = np.zeros_like(mu_induced)
        mu_sum = mu_induced_0.copy()
        for i in range(n_atoms_total):
            # for j in range(n_atoms_total):
            #     if i == j:
            #         continue
                # m = np.einsum('im,m->i', T_abij[i, j, 1:4, 0:1], M[j, 0:1])
                # m = np.einsum('im,m->i', T_abij[i, j, 1:4, :], M[j, :])
                # m = np.einsum('im,m->i', T_abij[i, j, 1:4, :], M[j, :])
                # m = np.einsum('im,m->i', T_abij[i, j, 1:4, :], M_induced[j, :])
                # m = np.einsum('im,m->i', T_abij[i, j, 1:4, :], M_induced[j, :])
                # m += np.einsum('im,m->i', T_abij[j, i, 1:4, :], M_induced[i, :])
                # m += np.einsum('mi,m->i', T_abij[i, j, :, 1:4], M_induced[j, :])
                
                # m += np.einsum('mi,m->i', T_abij[j, i, :, 1:4], M_induced[i, :]);
                # m += np.einsum('im,m->i', T_abij[i, j, 1:4, 0:1], M[j, 0:1])
                # m += np.einsum('im,m->i', T_abij[i, j, 1:4, 1:4], mu_induced_old[j])
                # m = np.einsum('im,m->i', T_abij[i, j, 1:4, 1:4], mu_induced_old[j])
            # mu_sum[i] += alpha_all[i] * m
            mu_sum_AB = np.einsum("bij,bj->i", T_abij[i, n_atoms_A:, 1:4, :], M_B_induced[:, :])
            mu_sum_BA = np.einsum("bji,aj->i", T_abij[n_atoms_A:, i, :, 1:4], M_A_induced[:, :])
            # mu_sum_AB = np.einsum("bij,bj->i", T_abij[i, n_atoms_A:, 1:4, :], M_B[:, :])
            # mu_sum_BA = np.einsum("bji,aj->i", T_abij[n_atoms_A:, i, :, 1:4], M_A[:, :])
            mu_sum[i] = (alpha_all[i] * (mu_sum_AB + mu_sum_BA))
            # tmp_B = np.einsum("b,aj->ab", alpha_B[i], M_A_induced[:, :])
            # mu_sum[i] += np.einsum('abij,ab->i', T_abij[i, :, 1:4, :], tmp_A)
            # mu_sum[i] += np.einsum('abij,ab->i', T_abij[i, :, 1:4, :], tmp_B)
            #     print(f"{i=}, {j=}, {mu_sum[i]=}")
            # print(f" i sum {i=}, {mu_sum[i]=}")
            M_induced[i, 1:4] = mu_sum[i]
        # mu_sum_A = np.einsum(
        #     # "n,abik,nk->ni", alpha_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B
        #     "n,abik,nk->ni", alpha_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B_induced
        # )
        # M_A_induced[:, 1:4] = mu_sum_A
        # mu_sum_B = np.einsum(
        #     "n,abki,nk->ni", alpha_B, T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], M_A_induced
        # )
        # M_B_induced[:, 1:4] = mu_sum_B
        # mu_sum = np.vstack([mu_sum_A, mu_sum_B])
        # print(f"{mu_sum = }")
        # mu_induced_0 = np.einsum(
        #     "n,abij,nj->ni", alpha_all, T_abij[:, :, 1:4, :], M_induced
        # )
        # print(f"{mu_induced_0 = }")
        # M_induced[:, 1:4] = mu_induced_0
        # mu_sum += mu_induced_0
        mu_induced = (1 - omega) * mu_induced_old + omega * (mu_sum)
        # print(f"Iteration {iteration + 1}:\n")
        mu_induced_diff = mu_induced - mu_induced_old
        # print(f"{mu_induced_diff}")

        # Check convergence
        delta = np.linalg.norm(mu_induced - mu_induced_old)
        if delta < convergence_threshold:
            print(f"   Converged after {iteration + 1} iterations.")
            break
    mu_induced_A = mu_induced[:n_atoms_A, :]
    mu_induced_B = mu_induced[n_atoms_A:, :]
    # print change in induced dipoles
    print('mu:')
    print(mu_all)
    print("mu(0):")
    print(mu_induced_old)
    print("mu(n):")
    print(mu_induced_0)
    # diff
    print(f"Change in induced dipoles over {iteration = }:")
    print(mu_induced_0 - mu_induced_old)

    # Calculate induction energy
    E_ind = 0.
    
    mu_0_A = np.einsum("abij,ai,bj->a", T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], mu_induced_0_A, M_B_induced) * constants.h2kcalmol   
    print(f"{mu_0_A = }")
    mu_0_B = np.einsum("abji,bi,bj->b", T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], mu_induced_0_B, M_A_induced) * constants.h2kcalmol 
    print(f"{mu_0_B = }")
    # E_0_ind = float(np.einsum("ai,abij,bj->", mu_induced_0_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B) + np.einsum("bi,abji,bj->", mu_induced_0_B, T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], M_A) * constants.h2kcalmol)
    E_0_ind = float(
        np.einsum("abij,ai,bj->", T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], mu_induced_0_A, M_B_induced) + 
        np.einsum("abji,bi,bj->", T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], mu_induced_0_B, M_A_induced)
    ) * constants.h2kcalmol
    print(f"{E_0_ind = }")

    mu_A = np.einsum("ai,abij,bj->a", mu_induced_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B) * constants.h2kcalmol   
    print(f"{mu_A = }")
    mu_B = np.einsum("bi,abji,bj->b", mu_induced_B, T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], M_A) * constants.h2kcalmol 
    print(f"{mu_B = }")
    # E_ind = np.einsum("ai,abij,bj->", mu_induced_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B) + np.einsum("bi,abji,bj->", mu_induced_B, T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], M_A)
    # E_ind = float(
    #     np.einsum("abij,ai,bj->", T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], mu_induced_0_A, M_B_induced) + 
    #     np.einsum("abji,bi,bj->", T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], mu_induced_0_B, M_A_induced)
    # )
    E_ind = float(
        # np.einsum("abij,ai,bj->", T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], mu_induced_A, M_B_induced) + 
        # np.einsum("abji,bi,bj->", T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], mu_induced_B, M_A_induced)
        np.einsum("abij,ai,bj->", T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], mu_induced_A, M_B) + 
        np.einsum("abji,bi,bj->", T_abij[:n_atoms_A, n_atoms_A:, :, 1:4], mu_induced_B, M_A)
    )
    # E_ind = np.einsum("ai,abij,bj->", mu_induced_A, T_AB, M_B) + np.einsum("bi,abji,bj->", mu_induced_B, T_BA, M_A)
    # E_ind += np.einsum("ni,abij,nj->", mu_induced_B, T_BA, M_A)
    # E_ind += np.einsum("ni,abij,nj->", mu_induced_0_B, T_BA, M_A)
    # E_ind += np.einsum("ni,abij,nj->", mu_induced_0_A, T_AB, M_B)
    # E_ind += np.einsum("ni,abji,nj->", mu_induced_0_B, T_BA, M_A)

    # E_ind += np.einsum("ni,baij,nj->", mu_induced_0_A, T_AB, M_B)
    # E_ind += np.einsum("ni,baji,nj->", mu_induced_0_B, T_BA, M_A)

    # E_ind += np.einsum("ni,abij,nj->", mu_induced_A, T_AB, M_B)
    # E_ind += np.einsum("ni,abji,nj->", mu_induced_B, T_BA, M_A)
    # E_ind += np.einsum("ni,abij,nj->", mu_induced, T_abij[:, :, 1:4, :], M_induced) / 2.0

    E_ind *= constants.h2kcalmol

    return E_ind, E_elst
