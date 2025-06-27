"""
Functions for evaluating electrostatics between atom-centered multipoles
"""

import numpy as np
from . import constants
import torch
from typing import Tuple


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
    dR_outer = torch.einsum('...i,...j->...ij', dR, dR)  # [N_A, N_B, 3, 3]
    
    # 3 * (dR ⊗ dR) - R^2 * δ
    T2_term1 = 3.0 * dR_outer  # [N_A, N_B, 3, 3]
    T2_term2 = R_squared.unsqueeze(-1).unsqueeze(-1) * delta  # [N_A, N_B, 3, 3]
    T2 = (T2_term1 - T2_term2) * R_inv_fifth.unsqueeze(-1).unsqueeze(-1)  # [N_A, N_B, 3, 3]
    
    # T3: Dipole-quadrupole interaction tensor [N_A, N_B, 3, 3, 3]
    R_inv_seventh = 1.0 / (R_safe**7)
    
    # Create Rdd tensor: dR_i * δ_jk for all pairs
    Rdd = torch.einsum('...i,jk->...ijk', dR, delta)  # [N_A, N_B, 3, 3, 3]
    
    # Create dR_i * dR_j * dR_k tensor
    dR_outer_outer = torch.einsum('...i,...j,...k->...ijk', dR, dR, dR)  # [N_A, N_B, 3, 3, 3]
    
    # Calculate T3
    T3_term1 = 15.0 * dR_outer_outer  # [N_A, N_B, 3, 3, 3]
    
    # Sum of permuted Rdd tensors
    # Rdd has shape [N_A, N_B, 3, 3, 3] with indices [batch_A, batch_B, i, j, k]
    # We want to permute the last 3 dimensions
    Rdd_sum = Rdd + Rdd.permute(0, 1, 3, 2, 4) + Rdd.permute(0, 1, 4, 3, 2)
    T3_term2 = 3.0 * R_squared.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * Rdd_sum
    
    T3 = -1.0 * (T3_term1 - T3_term2) * R_inv_seventh.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    # T4: Quadrupole-quadrupole interaction tensor [N_A, N_B, 3, 3, 3, 3]
    R_inv_ninth = 1.0 / (R_safe**9)
    
    # Create RRdd tensor: dR_i * dR_j * δ_kl
    # We need to expand delta to match batch dimensions
    delta_expanded = delta.view(1, 1, 3, 3).expand(N_A, N_B, 3, 3)
    RRdd = torch.einsum('...ij,...kl->...ijkl', dR_outer, delta_expanded)  # [N_A, N_B, 3, 3, 3, 3]
    
    # Create δδ tensor: δ_ij * δ_kl
    dddd = torch.einsum('ij,kl->ijkl', delta, delta)
    dddd_expanded = dddd.view(1, 1, 3, 3, 3, 3).expand(N_A, N_B, 3, 3, 3, 3)
    
    # Create dR_i * dR_j * dR_k * dR_l tensor
    dR_outer_outer_outer = torch.einsum('...ij,...kl->...ijkl', dR_outer, dR_outer)
    
    # Calculate T4
    T4_term1 = 105.0 * dR_outer_outer_outer
    
    # Sum of permuted RRdd tensors
    # RRdd has shape [N_A, N_B, 3, 3, 3, 3] with indices [batch_A, batch_B, i, j, k, l]
    # We need to permute the last 4 dimensions: i, j, k, l
    RRdd_sum = (
        RRdd +
        RRdd.permute(0, 1, 2, 4, 3, 5) +  # i,k,j,l
        RRdd.permute(0, 1, 2, 5, 4, 3) +  # i,l,k,j
        RRdd.permute(0, 1, 4, 3, 2, 5) +  # k,j,i,l
        RRdd.permute(0, 1, 5, 3, 4, 2) +  # l,j,k,i
        RRdd.permute(0, 1, 4, 5, 2, 3)    # k,l,i,j
    )
    
    T4_term2 = 15.0 * R_squared.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * RRdd_sum
    
    # Sum of permuted dddd tensors
    dddd_sum = (
        dddd_expanded +
        dddd_expanded.permute(0, 1, 2, 4, 3, 5) +  # i,k,j,l
        dddd_expanded.permute(0, 1, 2, 5, 4, 3)    # i,l,k,j
    )
    
    T4_term3 = 3.0 * (R_squared**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dddd_sum
    
    T4 = (T4_term1 - T4_term2 + T4_term3) * R_inv_ninth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
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
    mol_dimer, qA, muA, thetaA, qB, muB, thetaB, ensure_traceless=False,
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
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j, ensure_traceless=ensure_traceless
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
    RA, qA, muA, thetaA, RB, qB, muB, thetaB, ensure_traceless=True
):
    T0, T1, T2, T3, T4 = T_cart(RA, RB)

    # Most inputs will already be traceless, but we can ensure this is the case
    if ensure_traceless:
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
    # print(pair_mat.shape)
    # QA[:,0] -= maskA * np.sum(QA[:,0]) / np.sum(maskA)
    # QB[:,0] -= maskB * np.sum(QB[:,0]) / np.sum(maskB)
    # QA[:,0] -= np.average(QA[:,0])
    # QB[:,0] -= np.average(QB[:,0])
    # print(f'{np.sum(QA[:,0]):.2f} {np.sum(QB[:,0]):.2f}')
    # print(QA[:,0], QB[:,0])

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

            # print('pair', pair_energy)
            total_energy += pair_energy
            pair_mat[ia][ib] = pair_energy

    Har2Kcalmol = 627.5094737775374055927342256

    return total_energy * Har2Kcalmol, pair_mat * Har2Kcalmol


# def eval_dimer(moldenA, moldenB, h5A, h5B, print_=False):
#
#    # Keep R in a.u. (molden convention)
#    RA, ZA = proc_molden(moldenA)
#    RB, ZB = proc_molden(moldenB)
#
#    # Get horton output
#    hfA = h5py.File(h5A, 'r')
#    hfB = h5py.File(h5B, 'r')
#
#    # get multipoles
#    QA = hfA['cartesian_multipoles'][:]
#    QB = hfB['cartesian_multipoles'][:]
#
#    total_energy = 0.0
#    pair_energies_l = []
#
#    # calculate multipole electrostatics for each atom pair
#    for ia in range(len(ZA)):
#        for ib in range(len(ZB)):
#            rA = RA[ia]
#            zA = ZA[ia]
#            qA = QA[ia]
#
#            rB = RB[ib]
#            zB = ZB[ib]
#            qB = QB[ib]
#
#            #print(zA, qA[0], zB, qB[0])
#
#            ## [E0 (qq), E1 (qmu), E2(qTh+mumu), E3(muTh), E4(ThTh)]
#            #pair_energies = eval_interaction(rA, qA[0], qA[1:4], (3.0/2.0) * qpole_redundant(qA[4:10]),
#            #        rB, qB[0], qB[1:4], (3.0/2.0) * qpole_redundant(qB[4:10]))
#
#            # [E0 (qq), E1 (qmu), E2(qTh+mumu), E3(muTh), E4(ThTh)]
#            pair_energies = eval_interaction_damp(rA, zA, qA[0] - zA, qA[1:4], (3.0/2.0) * qpole_redundant(qA[4:10]),
#                    rB, zB, qB[0] - zB, qB[1:4], (3.0/2.0) * qpole_redundant(qB[4:10]))
#            pair_energy = np.sum(pair_energies)
#            pair_energies_l.append(pair_energies)
#
#            total_energy += pair_energy
#
#    pair_energies_l = np.array(pair_energies_l)
#    Har2Kcalmol = 627.509
#
#    if print_:
#        print('  Energies (mEh)')
#        print('     E_q        E_u        E_Q        Total')
#        for l in pair_energies_l:
#            print(f'{l[0]*Har2Kcalmol:10.5f} {l[1]*Har2Kcalmol:10.5f} {l[2]*Har2Kcalmol:10.5f} {np.sum(l)*Har2Kcalmol:10.5f}')
#        l = np.sum(pair_energies_l, axis=0)
#        print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#        print(f'{l[0]*Har2Kcalmol:10.5f} {l[1]*Har2Kcalmol:10.5f} {l[2]*Har2Kcalmol:10.5f} {np.sum(l)*Har2Kcalmol:10.5f}')
#
#    E_q = np.sum(pair_energies_l[:,:1])
#    E_qu = np.sum(pair_energies_l[:,:2])
#    E_quQ = np.sum(pair_energies_l[:,:3])
#    return E_q, E_qu, E_quQ


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_pickle("../directional-mpnn/data/HBC6_hfjdz.pkl")
    df_prd = pd.read_pickle("../directional-mpnn/preds/HBC6_hfjdz_modelB.pkl")

    print(df.cartesian_multipoles[0].shape)
    print(df_prd.cartesian_multipoles_prd[0].shape)
    print(df.columns)

    df_elst = {
        "name": [],
        "dimer": [],
        "dist": [],
        "e_mbis_q": [],
        "e_mbis_u": [],
        "e_mbis_Q": [],
        "e_mpnn_q": [],
        "e_mpnn_u": [],
        "e_mpnn_Q": [],
        "e_sapt": [],
    }

    errs = []
    for i in range(0, len(df.index), 2):
        print(i, df.name[i])
        RA = df.R[i]
        RB = df.R[i + 1]
        ZA = df.Z[i]
        ZB = df.Z[i + 1]
        QA = df.cartesian_multipoles[i]
        QB = df.cartesian_multipoles[i + 1]
        QA_prd = df_prd.cartesian_multipoles_prd[i]
        QB_prd = df_prd.cartesian_multipoles_prd[i + 1]
        # print(QA[:,0]-QA_prd[:,0])
        # print(QB[:,0]-QB_prd[:,0])
        e_mbis = eval_dimer(
            RA,
            RB,
            ZA,
            ZB,
            QA,
            QB,
        )
        e_mpnn = eval_dimer(RA, RB, ZA, ZB, QA_prd, QB_prd)
        errs.append(e_mbis - e_mpnn)
        print(
            f"{df.sapt0_hfjdz_elst[i] * 627.509:8.3f} {e_mbis * 627.509:8.3f} {e_mpnn * 627.509:8.3f}"
        )
        print(df.name[i].split("-"))
    errs = np.array(errs)
    print(np.average(np.abs(errs)) * 627.509)

    # df_elst = pd.DataFrame.from_dict(data=df_elst)
    # df_elst.to_pickle(f'preds/HBC6_elst.pkl', protocol=4)
    # print(df_elst)

    # older comment vvv

    # dfA = pd.read_pickle('data/S66x8-A.pkl')
    # dfA_prd = pd.read_pickle('preds/S66x8-A_modelB.pkl')
    # dfB = pd.read_pickle('data/S66x8-B.pkl')
    # dfB_prd = pd.read_pickle('preds/S66x8-B_modelB.pkl')

    # for i in range(66*8):

    #    if i % 8 == 0:
    #        print()
    #        print(dfA.name[i])
    #    RA = dfA.R[i]
    #    RB = dfB.R[i]
    #    ZA = dfA.Z[i]
    #    ZB = dfB.Z[i]
    #    QA = dfA.cartesian_multipoles[i]
    #    QB = dfB.cartesian_multipoles[i]
    #    QA_prd = dfA_prd.cartesian_multipoles_prd[i]
    #    QB_prd = dfB_prd.cartesian_multipoles_prd[i]
    #    #print(QA[:,0]-QA_prd[:,0])
    #    #print(QB[:,0]-QB_prd[:,0])
    #    _,_,e_mbis = eval_dimer(RA, RB, ZA, ZB, QA, QB, print_=False)
    #    _,_,e_mpnn = eval_dimer(RA, RB, ZA, ZB, QA_prd, QB_prd, print_=False)
    #    print(f'{e_mbis * 627.509:8.3f} {e_mpnn * 627.509:8.3f}')
