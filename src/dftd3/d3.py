import qcelemental 
import apnet_pt
import torch
import os
h2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstrom = qcelemental.constants.bohr2angstroms

from data import radii, r4r2
from rational import rational_damping
from weights import weight_references 
import defaults

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


def exp_count(
    distances: torch.tensor, 
    cov_r: torch.tensor, 
) -> torch.tensor:
    
    k2 = 4.0 / 3.0 #ad hoc factor so the cn is reasonable for molecules
    k1 = 16 #large so distant atoms are not counted so CN does not depend on size of system
    
    return 1.0 / (1.0 + torch.exp(-k1 * (torch.divide(k2 * cov_r, distances) - 1.0)))

def cn_d3_intermolecular(
    batch,
) -> torch.tensor:
    
    RA = batch.RA
    dd = {"device": RA.device, "dtype": RA.dtype}

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
    param: dict[str, torch.tensor],
    **kwargs,

):
    RA = batch.RA
    dd = {"device": RA.device, "dtype": RA.dtype}

    path = os.path.join(os.path.dirname(__file__), "data/reference-c6.pt")
    kwargs = {"weights_only" : True, "map_location" : dd['device']}
    print(torch.load(path, **kwargs))
    ref_c6 = torch.load(path, **kwargs).type(dtype=dd['dtype'])

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

    
    weights_A = weight_references(ZA, cn_A,)
    weights_B = weight_references(ZB, cn_B,)

    rc6 = ref_c6[ZA, ZB]
    c6 = torch.einsum("ijk,ij,ik->i", rc6, weights_A, weights_B)
    distances, _ = get_distances(RA=RA, RB=RB, e_source=e_source_full, e_target=e_target_full)

    #C8 is computed recursively from c6

    #Q_A = sqrt(Z) * r^4/r^2
    r4_over_r2 = r4r2.R4R2(**dd)
    #ad hoc nuclear charge dependent factor
    sqrtz = torch.sqrt(
        torch.arange(len(r4_over_r2), **dd)
    )
    Q = r4_over_r2 * sqrtz
    #C_8 = 3 * C_6 * sqrt(Q_A * Q_B)

    #quotient of C8 and C6, used later by damping function
    qAqB = 3 * torch.sqrt((Q[ZA] * Q[ZB]))
    c8 = c6 * qAqB

    #c8 are not environment aware
    print(f"{c8 = }")

   
    t6 = rational_damping(6, distances, qAqB, param,)
    t8 = rational_damping(8, distances, qAqB, param,)
    
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