import pytest
import apnet_pt
import torch
import qcelemental as qcel
import os
import numpy as np

lr_water_dimer = qcel.models.Molecule.from_data("""
0 1
--
0 1
O                    -1.326958230000    -0.105938530000     0.018788150000
H                    -1.931665240000     1.600174320000    -0.021710520000
H                     0.486644280000     0.079598090000     0.009862480000
--
0 1
O                     8.088671270000     0.019951580000    -0.007942850000
H                     8.800382980000    -0.808466680000     1.439822410000
H                     8.792148880000    -0.879960520000    -1.416549430000
units bohr
""")

file_dir = os.path.dirname(os.path.abspath(__file__))

mol_dimer_big = qcel.models.Molecule.from_data(
    """
    1 1
    C       0.0545060001    -0.1631290019   -1.1141539812
    C       -0.9692260027   -1.0918780565   0.6940879822
    C       0.3839910030    0.5769280195    -0.0021170001
    C       1.3586950302    1.7358809710    0.0758149996
    N       -0.1661809981   -0.0093130004   1.0584640503
    N       -0.8175240159   -1.0993789434   -0.7090409994
    H       0.3965460062    -0.1201139987   -2.1653149128
    H       -1.5147459507   -1.6961929798   1.3000769615
    H       0.7564010024    2.6179349422    0.4376020133
    H       2.2080008984    1.5715960264    0.7005280256
    H       1.7567750216    2.0432629585    -0.9004560113
    H       -0.1571149975   0.2784340084    1.9974440336
    H       -1.2523859739   -1.9090379477   -1.2904200554
    --
    -1 1
    C       -5.6793351173   2.6897408962    7.4496979713
    C       -4.5188479424   3.5724110603    6.9706201553
    N       -6.1935510635   1.6698499918    6.8358440399
    N       -6.2523350716   2.9488639832    8.6100416183
    N       -7.1709971428   1.1798499823    7.7206158638
    N       -7.2111191750   1.9820170403    8.7515516281
    H       -4.9275932312   4.5184249878    6.4953727722
    H       -3.8300020695   3.8421258926    7.6719899178
    H       -4.1228170395   3.0444390774    6.1303391457
    units angstrom
                """
)

def test_elst_multipoles():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols(
        [monA, monB, monA.copy(), monB.copy()], batch_size=3
    )
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    E_elst = apnet_pt.multipole.eval_qcel_dimer(
        mol_dimer=lr_water_dimer,
        qA=mtp_A[0].numpy(),
        muA=mtp_A[1].numpy(),
        thetaA=mtp_A[2].numpy(),
        qB=mtp_B[0].numpy(),
        muB=mtp_B[1].numpy(),
        thetaB=mtp_B[2].numpy(),
    )
    print(f"E_elst = {E_elst:.6f} kcal/mol")
    E_ref = -0.853646
    assert abs(E_elst - E_ref) < 1e-6, f"Expected {E_ref}, got {E_elst}"


def test_elst_charge_dipole_qpole():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols(
        [monA, monB, monA.copy(), monB.copy()], batch_size=3
    )
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    E_q, E_dp, E_qpole = apnet_pt.multipole.eval_qcel_dimer_individual(
        mol_dimer=lr_water_dimer,
        qA=mtp_A[0].numpy(),
        muA=mtp_A[1].numpy(),
        thetaA=mtp_A[2].numpy(),
        qB=mtp_B[0].numpy(),
        muB=mtp_B[1].numpy(),
        thetaB=mtp_B[2].numpy(),
    )
    print(f"E_q = {E_q:.6f} kcal/mol")
    print(f"E_dp = {E_dp:.6f} kcal/mol")
    print(f"E_qpole = {E_qpole:.6f} kcal/mol")
    E_q_ref = -1.239722
    E_dp_ref = 0.392898
    E_qpole_ref = -0.006823
    assert abs(E_q - E_q_ref) < 1e-6, f"Expected {E_q_ref}, got {E_q}"
    assert abs(E_dp - E_dp_ref) < 1e-6, f"Expected {E_dp_ref}, got {E_dp}"
    assert abs(E_qpole - E_qpole_ref) < 1e-6, f"Expected {E_qpole_ref}, got {E_qpole}"


def test_elst_multipoles_am_hirshfeld():
    atom_model = apnet_pt.AtomModels.ap3_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    atom_model.set_pretrained_model(file_dir + "/../models/am_hf_ensemble/am_0.pt")
    print(atom_model)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols(
        [monA, monB, monA.copy(), monB.copy()], batch_size=3
    )
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    E_elst = apnet_pt.multipole.eval_qcel_dimer(
        mol_dimer=lr_water_dimer,
        qA=mtp_A[0].numpy(),
        muA=mtp_A[1].numpy(),
        thetaA=mtp_A[2].numpy(),
        qB=mtp_B[0].numpy(),
        muB=mtp_B[1].numpy(),
        thetaB=mtp_B[2].numpy(),
    )
    print(f"E_elst = {E_elst:.6f} kcal/mol")
    E_ref = -0.7430384309295008
    assert abs(E_elst - E_ref) < 1e-6, f"Expected {E_ref}, got {E_elst}"


def test_ap2_elst_multipoles():
    am = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
            
    ap2 = apnet_pt.APNet2Model(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
        atom_model=am.model,
    )
    mol_dimer = mol_dimer_big
    batch = ap2.example_input(mol_dimer, r_cut=1000.0, r_cut_im=1000.0)
    print(batch)
    print(batch.qA, batch.qB)
    dR_sr, dR_sr_xyz = ap2.model.get_distances(
        batch.RA, batch.RB, batch.e_ABsr_source, batch.e_ABsr_target
    )
    Elst = ap2.model.mtp_elst(
        qA=batch.qA,
        muA=batch.muA,
        quadA=batch.quadA,
        qB=batch.qB,
        muB=batch.muB,
        quadB=batch.quadB,
        e_ABsr_source=batch.e_ABsr_source,
        e_ABsr_target=batch.e_ABsr_target,
        dR_ang=dR_sr,
        dR_xyz_ang=dR_sr_xyz,
    )
    Elst = torch.sum(Elst)
    print(f"Elst = {Elst:.6f} kcal/mol")
    # Reference value from 
    E_ref = apnet_pt.multipole.eval_qcel_dimer(
        mol_dimer=mol_dimer,
        qA=batch.qA.numpy(),
        muA=batch.muA.numpy(),
        thetaA=batch.quadA.numpy(),
        qB=batch.qB.numpy(),
        muB=batch.muB.numpy(),
        thetaB=batch.quadB.numpy(),
    )
    print(f"E_ref = {E_ref:.6f} kcal/mol")
    assert abs(Elst - E_ref) < 1e-6, f"Expected {E_ref}, got {Elst}"


if __name__ == "__main__":
    # test_elst_multipoles()
    # test_elst_multipoles_am_hirshfeld()
    # test_elst_charge_dipole_qpole()
    test_ap2_elst_multipoles()
