
#Tad-dftd3 imports 

#rational damping I think now has no dependence on mctc, it just has the typing stuff
from tad_dftd3.damping import rational_damping
from tad_dftd3.reference import Reference
from tad_dftd3 import data, defaults, model
from tad_dftd3.typing import (
    DD,
    CountingFunction,
    DampingFunction,
    Tensor,
    WeightingFunction,
)


#tad_mctc imports that I copied over
from cn import radii
from cn.count import exp_count 

from rational import rational_damping
import qcelemental 
import apnet_pt
import torch
import numpy as np
h2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstrom = qcelemental.constants.bohr2angstroms


param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
}

water_water_dimer = qcelemental.models.Molecule.from_data("""
0 1
--
0 1
O                    -1.326958230000    -0.105938530000     0.018788150000
H                    -1.931665240000     1.600174320000    -0.021710520000
H                     0.486644280000     0.079598090000     0.009862480000
--
0 1
O                     4.287563290000     0.049775580000     0.000960040000
H                     4.999275000000    -0.778642690000     1.448725300000
H                     4.991040900000    -0.850136520000    -1.407646550000
units bohr
no_com
no_reorient
""")

mols = [
    water_water_dimer
]

batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
    [
        apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
            mol, r_cut=5.0, dimer_ind=n, r_cut_im=torch.inf
        )
        for n, mol in enumerate(mols)
    ]
)

def get_distances(RA, RB, e_source, e_target):
        RA_source = RA.index_select(0, e_source)
        RB_target = RB.index_select(0, e_target)
        dR_xyz = RB_target - RA_source

        # Compute distances with safe operation for square root
        # dR = torch.sqrt(nn.functional.relu(torch.sum(dR_xyz**2, dim=-1)))
        dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        return dR, dR_xyz


def cn_d3_intermolecular(
    batch,
    *,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
) -> Tensor:
    """
    Compute the D3 fractional coordination (exponential counting function).

    Parameters
    ----------
    batch : AP2 Fused DS
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``,
        the steepness of the counting function, which defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.
    
    Returns
    -------
    Tensor
         

    Raises
    ------
    ValueError NEED TO IMPLEMENT
        If shape mismatch between ``numbers``, ``positions`` and
        ``rcov`` is detected.
    """
    RA = batch.RA

    dd: DD = {"device": RA.device, "dtype": RA.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.D3_CN_CUTOFF, **dd)


    e_source_full = torch.concatenate([batch.e_ABsr_source, batch.e_ABlr_source,])
    e_target_full = torch.concatenate([batch.e_ABsr_target, batch.e_ABlr_target,])
    
    ZA = batch.ZA
    ZB = batch.ZB
    RA = batch.RA
    RB = batch.RB

    ZA = ZA.index_select(0, e_source_full)
    ZB = ZB.index_select(0, e_target_full)
    RA = RA.index_select(0, e_source_full)
    RB = RB.index_select(0, e_target_full)

    #Not really just the sum of the covalent radii has a scale factor of 4/3 applied
    rcov = radii.COV_D3(**dd)[ZA] + radii.COV_D3(**dd)[ZB] 
    print(f"{rcov = }")
    
    distances, _ = get_distances(RA, RB, e_source_full, e_target_full)
    cn = torch.where(
        (distances <= cutoff),
        exp_count(distances, rcov),
        torch.tensor(0.0, **dd)
    )

    size = e_source_full.max().item() + 1
    cn_A = torch.zeros(size, dtype=cn.dtype)
    cn_A.scatter_reduce_(0, e_source_full, cn, reduce="sum", include_self=False)
    
    cn_B = torch.zeros(size, dtype=cn.dtype)
    cn_B.scatter_reduce_(0, e_target_full, cn, reduce="sum", include_self=False)
    return cn_A, cn_B


def apnet_dispersion_batch(
    batch,
    param: dict[str, Tensor],
    *,
    ref: Reference | None = None,
    r4r2: Tensor | None = None,
    cutoff: Tensor | None = None,
    weighting_function: WeightingFunction = model.gaussian_weight,
    **kwargs,

):
    RA = batch.RA
    dd: DD = {"device": RA.device, "dtype": RA.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.D3_DISP_CUTOFF, **dd)
    if ref is None:
        ref = Reference(**dd)

    cn_A, cn_B = cn_d3_intermolecular(
        batch,
    ) 

    ZA = batch.ZA
    RA = batch.RA / bohr2angstrom

    ZB = batch.ZB
    RB = batch.RB / bohr2angstrom
    
    e_source_full = torch.concatenate([batch.e_ABsr_source, batch.e_ABlr_source,])
    e_target_full = torch.concatenate([batch.e_ABsr_target, batch.e_ABlr_target,])
    cn_A = cn_A.index_select(0, e_source_full)

    cn_B = cn_B.index_select(0, e_target_full)    
    ZA = ZA.index_select(0, e_source_full)
    ZB = ZB.index_select(0, e_target_full)

    weights_A = model.weight_references(ZA, cn_A, ref, weighting_function)
    weights_B = model.weight_references(ZB, cn_B, ref, weighting_function)
    
    rc6 = ref.c6[ZA, ZB]
    c6 = torch.einsum("ijk,ij,ik->i", rc6, weights_A, weights_B)
    distances, _ = get_distances(RA=RA, RB=RB, e_source=e_source_full, e_target=e_target_full)

    #C8 is computed recursively from c6

    #Q_A = sqrt(Z) * r^4/r^2
    r4r2 = data.R4R2_alt(**dd)
    #ad hoc nuclear charge dependent factor
    sqrtz = torch.sqrt(
        torch.arange(len(r4r2), **dd)
    )
    Q = r4r2 * sqrtz
    #C_8 = 3 * C_6 * sqrt(Q_A * Q_B)

    #quotient of C8 and C6, used later by damping function
    qAqB = 3 * torch.sqrt((Q[ZA] * Q[ZB]))
    c8 = c6 * qAqB

    #c8 are not pair specific or environment aware, that's why you see duplicates
    print(f"{c8 = }")
    #c8 = tensor([340.0482, 116.2165, 116.2165, 116.2165,  44.5324,  44.5324, 116.2165, 44.5324,  44.5324])

   
    t6 = rational_damping(6, distances, qAqB, param, **kwargs)
    t8 = rational_damping(8, distances, qAqB, param, **kwargs)
    
    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))
    e6 = -1 * (c6 * t6) * s6
    e8 = -1 * (c8 * t8) * s8
    pairwise_energies = e6 + e8
    print(pairwise_energies * h2kcalmol)
    #The pairwise energies look like this
    #tensor([-0.3419, -0.0909, -0.0910, -0.1002, -0.0301, -0.0301, -1.3880, -0.4003, -0.4006])
    #The -1.3880 is the dispersion for your closest contact atoms

    #These are the pairwise energies from simple dftd3
    #pairwise_simple_Es= tensor([-0.2734, -0.0599, -0.8294, -0.0539, -0.0127, -0.1683, -0.0539, -0.0127, -0.1684], dtype=torch.float64)
    #The doesn't correlate at all with the pairwise energies from the intermolecular approach here
    #Well, okay the third one that is pretty similar in magnitude, -0.8924 to the -1.3880 one
    print(torch.sum(pairwise_energies) * h2kcalmol)
    
    return

torch.set_printoptions(sci_mode=False)
cn_A, cn_B = cn_d3_intermolecular(batch)
apnet_dispersion_batch(batch, param)
print(f"{cn_A = }")
print(f"{cn_B = }")