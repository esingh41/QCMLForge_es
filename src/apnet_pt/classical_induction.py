import torch
from torchmin import minimize  # use torchmin for minimization


def set_constants():
    M_PI = 3.14159265358979323846
    E_CHARGE = 1.602176634e-19
    AVOGADRO = 6.02214076e23
    EPSILON0 = 1e-6 * 8.8541878128e-12 / (E_CHARGE * E_CHARGE * AVOGADRO)
    return 1 / (4 * M_PI * EPSILON0)


ONE_4PI_EPS0 = set_constants()


def get_displacements(r_core, r_shell):
    """
    Given initial positions of a crystal structure or trajectory file,
    initialize shell charge site positions and charges.
    """
    # Compute norm along last dimension and create a boolean mask for nonzero shells
    shell_mask = torch.norm(r_shell, dim=-1) > 0.0
    d = r_core - r_shell
    # Expand mask to last dimension
    d = torch.where(
        shell_mask.unsqueeze(-1),
        d,
        torch.tensor(
            0.0,
            dtype=d.dtype,
            device=d.device,
        ),
    )
    return d


def Upol(Dij, k):
    """
    Calculates polarization energy,
        U_pol = 1/2 Σ k_i * ||d_i||^2.
    """
    d_mag = torch.norm(Dij, dim=2)
    return 0.5 * torch.sum(k * d_mag**2)


def Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale):
    """
    Computes the Coulomb (electrostatic) contribution to the induction energy.
    """
    # Expand Dij to allow broadcasting:
    # Di: (nmol, 1, natoms, 1, 3) and Dj: (1, nmol, 1, natoms, 3)
    Di = Dij.unsqueeze(1).unsqueeze(3)  # Dij[:, None, :, None, :]
    Dj = Dij.unsqueeze(0).unsqueeze(2)  # Dij[None, :, None, :, :]

    Rij_norm = torch.norm(Rij, dim=-1)
    Rij_Di_norm = torch.norm(Rij + Di, dim=-1)
    Rij_Dj_norm = torch.norm(Rij - Dj, dim=-1)
    Rij_Di_Dj_norm = torch.norm(Rij + Di - Dj, dim=-1)

    # Replace zero norms with infinity to avoid division by zero.
    _Rij_norm = torch.where(
        Rij_norm == 0.0, torch.tensor(
            float("inf"), dtype=Rij_norm.dtype), Rij_norm
    )
    _Rij_Di_norm = torch.where(
        Rij_Di_norm == 0.0,
        torch.tensor(float("inf"), dtype=Rij_Di_norm.dtype),
        Rij_Di_norm,
    )
    _Rij_Dj_norm = torch.where(
        Rij_Dj_norm == 0.0,
        torch.tensor(float("inf"), dtype=Rij_Dj_norm.dtype),
        Rij_Dj_norm,
    )
    _Rij_Di_Dj_norm = torch.where(
        Rij_Di_Dj_norm == 0.0,
        torch.tensor(float("inf"), dtype=Rij_Di_Dj_norm.dtype),
        Rij_Di_Dj_norm,
    )

    Sij = 1.0 - (1.0 + 0.5 * Rij_norm * u_scale) * \
        torch.exp(-u_scale * Rij_norm)
    Sij_Di = 1.0 - (1.0 + 0.5 * Rij_Di_norm * u_scale) * torch.exp(
        -u_scale * Rij_Di_norm
    )
    Sij_Dj = 1.0 - (1.0 + 0.5 * Rij_Dj_norm * u_scale) * torch.exp(
        -u_scale * Rij_Dj_norm
    )
    Sij_Di_Dj = 1.0 - (1.0 + 0.5 * Rij_Di_Dj_norm * u_scale) * torch.exp(
        -u_scale * Rij_Di_Dj_norm
    )

    U_coul = (
        Qi_core * Qj_core / _Rij_norm
        + Qi_shell * Qj_core / _Rij_Di_norm
        + Qi_core * Qj_shell / _Rij_Dj_norm
        + Qi_shell * Qj_shell / _Rij_Di_Dj_norm
    )

    U_coul_intra = (
        Sij * (-Qi_shell) * (-Qj_shell) / _Rij_norm
        + Sij_Di * (Qi_shell) * (-Qj_shell) / _Rij_Di_norm
        + Sij_Dj * (-Qi_shell) * (Qj_shell) / _Rij_Dj_norm
        + Sij_Di_Dj * (Qi_shell) * (Qj_shell) / _Rij_Di_Dj_norm
    )

    # Remove self-interactions by applying appropriate masks.
    I_intra = (
        torch.eye(
            U_coul_intra.shape[0], dtype=U_coul_intra.dtype, device=U_coul_intra.device
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    I_self = (
        torch.eye(
            U_coul_intra.shape[-1], dtype=U_coul_intra.dtype, device=U_coul_intra.device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    U_coul_intra = (U_coul_intra * I_intra) * (1 - I_self)
    U_coul_intra = 0.5 * torch.sum(
        torch.where(
            torch.isfinite(U_coul_intra),
            U_coul_intra,
            torch.tensor(0.0, dtype=U_coul_intra.dtype,
                         device=U_coul_intra.device),
        )
    )

    I = (
        torch.eye(U_coul.shape[0], dtype=U_coul.dtype, device=U_coul.device)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    U_coul_inter = U_coul * (1 - I)
    U_coul_inter = 0.5 * torch.sum(
        torch.where(
            torch.isfinite(U_coul_inter),
            U_coul_inter,
            torch.tensor(0.0, dtype=U_coul.dtype, device=U_coul.device),
        )
    )
    return ONE_4PI_EPS0 * (U_coul_inter + U_coul_intra)


def Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape=None):
    """
    Calculates the total induction energy:
        U_ind = U_pol + U_coul.
    """
    if reshape is not None:
        Dij = Dij.view(reshape)
    U_pol = Upol(Dij, k)
    U_coul_val = Ucoul(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale)
    print(f"U_pol: {U_pol:.4f}, U_coul: {U_coul_val:.4f}")
    return U_pol + U_coul_val


def drudeOpt(
    Rij,
    Dij0,
    Qi_shell,
    Qj_shell,
    Qi_core,
    Qj_core,
    u_scale,
    k,
    methods=["BFGS"],
    d_ref=None,
    reshape=None,
):
    """
    Optimizes the Drude (core/shell) displacements by minimizing the induction energy U_ind.
    """

    def Uind_min(Dij):
        return Uind(Rij, Dij, Qi_shell, Qj_shell, Qi_core, Qj_core, u_scale, k, reshape)

    # Use torchmin for minimization (here, method is assumed to be 'BFGS')
    res = minimize(Uind_min, x0=Dij0, method="BFGS")
    d_opt = res["x"]
    if reshape is not None:
        d_opt = d_opt.view(reshape)
    if d_ref is not None:
        diff = torch.norm(d_ref - d_opt)
    return d_opt

def compute_drude_oscillator_U_ind(
    Dij,
    Rij,
    Qi_shell,
    Qj_shell,
    Qi_core,
    Qj_core,
    u_scale,
    k,
):
    # Get inputs (all arrays are now PyTorch tensors)
    # Flatten Dij to a 1D tensor for minimization and later reshape it back.
    Dij_flat = Dij.view(-1)
    Dij_opt = drudeOpt(
        Rij,
        Dij_flat,
        Qi_shell,
        Qj_shell,
        Qi_core,
        Qj_core,
        u_scale,
        k,
        reshape=Dij.shape,
    )
    U_ind = Uind(Rij, Dij_opt, Qi_shell, Qj_shell,
                 Qi_core, Qj_core, u_scale, k)
    return U_ind
