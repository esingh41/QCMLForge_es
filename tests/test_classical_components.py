import pytest
import apnet_pt
import qcelemental as qcel
import os
import pandas as pd
from pprint import pprint
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


def test_elst_multipoles():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols([monA, monB, monA.copy(), monB.copy()], batch_size=3)
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
    multipoles = atom_model.predict_qcel_mols([monA, monB, monA.copy(), monB.copy()], batch_size=3)
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


def test_elst_charge_dipole_qpole_pairwise():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols([monA, monB, monA.copy(), monB.copy()], batch_size=3)
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=lr_water_dimer,
        qA=mtp_A[0].numpy(),
        muA=mtp_A[1].numpy(),
        thetaA=mtp_A[2].numpy(),
        qB=mtp_B[0].numpy(),
        muB=mtp_B[1].numpy(),
        thetaB=mtp_B[2].numpy(),
    )
    print(f"Total energy = {total_energy:.6f} kcal/mol")
    print(f"E_qqs = {E_qqs.sum():.6f} kcal/mol")
    print(f"E_qus = {E_qus.sum():.6f} kcal/mol")
    print(f"E_uus = {E_uus.sum():.6f} kcal/mol")
    print(f"E_qQs = {E_qQs.sum():.6f} kcal/mol")
    print(f"E_uQs = {E_uQs.sum():.6f} kcal/mol")
    print(f"E_QQs = {E_QQs.sum():.6f} kcal/mol")
    return 


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
    multipoles = atom_model.predict_qcel_mols([monA, monB, monA.copy(), monB.copy()], batch_size=3)
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



def test_induced_dipole():
    # check here for CLIFF eval: /home/awallace43/projects/multipoles/cliff_tests
    df = pd.read_pickle(file_dir + os.sep + os.path.join("dataset_data", "water_dimer_pes2.pkl"))
    print(df[['cliff_elst', 'SAPT0 ELST ENERGY adz']])
    print(df[['cliff_exch', 'SAPT0 EXCH ENERGY adz']])
    print(df[['cliff_indu', 'SAPT0 IND ENERGY adz']])
    print(df[['cliff_disp', 'SAPT0 DISP ENERGY adz']])
    for n, r in df.iterrows():
        sapt0_ind = r['SAPT0 IND ENERGY adz']
        sapt0_elst = r['SAPT0 ELST ENERGY adz']
        mol = r['qcel_molecule']
        # qm_tools_aw.molecular_visualization.visualize_molecule(
        #     mol,
        #    temp_filename=f"{n}_water_dimer_sapt0_ind.html",
        #                                                        )
        # Distance between monomers
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(np.sum((monA.geometry[:, None] - monB.geometry)**2, axis=2)).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r['q_A pbe0/atz']
        muA = r['mu_A pbe0/atz']
        thetaA = r['theta_A pbe0/atz']
        qB = r['q_B pbe0/atz']
        muB = r['mu_B pbe0/atz']
        thetaB = r['theta_B pbe0/atz']
        vrA = r['vol_ratios_A pbe0/atz']
        vrB = r['vol_ratios_B pbe0/atz']
        vwA = r['val_widths_A pbe0/atz']
        vwB = r['val_widths_B pbe0/atz']
        total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs = apnet_pt.multipole.eval_qcel_dimer_individual_components(
            mol_dimer=mol,
            qA=qA,
            muA=muA,
            thetaA=thetaA,
            qB=qB,
            muB=muB,
            thetaB=thetaB,
        )
        E_qq = E_qqs.sum()
        E_qu = E_qus.sum()
        E_uu = E_uus.sum()
        E_qQ = E_qQs.sum()
        E_uQ = E_uQs.sum()
        print(f"{total_energy=:.6f} kcal/mol")
        print(f"{E_qq=:.6f} kcal/mol")
        print(f"{E_qu=:.6f} kcal/mol")
        print(f"{E_uu=:.6f} kcal/mol")
        print(f"{E_qQ=:.6f} kcal/mol")
        print(f"{E_uQ=:.6f} kcal/mol")
        induction_energy = apnet_pt.multipole.dimer_induced_dipole(
            mol,
            qA=qA,
            muA=muA,
            thetaA=thetaA,
            qB=qB,
            muB=muB,
            thetaB=thetaB,
            hirshfeld_volume_ratio_A=vrA,
            hirshfeld_volume_ratio_B=vrB,
            valence_widths_A=vwA,
            valence_widths_B=vwB,
        )
        print(f"Distance between monomers: {dist * bohr2angstrom:.2f} A")
        print(f"CLIFF elst       = {r['cliff_elst']:.6f}")
        print(f"SAPT elst        = {sapt0_elst:.6f} kcal/mol")
        print(f"SAPT induction   = {sapt0_ind:.6f} kcal/mol")
        print(f"Induction energy = {induction_energy:.6f} kcal/mol")
        print(f"CLIFF induction  = {r['cliff_indu']:.6f}")

def test_induced_dipole_bz_meoh():
    df = pd.read_pickle(file_dir + os.sep + os.path.join("dataset_data", "df_bz_meoh_mbis.pkl"))
    for n, r in df.iterrows():
        sapt0_ind = r['SAPT0 IND ENERGY adz']
        sapt0_elst = r['SAPT0 ELST ENERGY adz']
        mol = r['qcel_molecule']
        # qm_tools_aw.molecular_visualization.visualize_molecule(
        #     mol,
        #    temp_filename=f"{n}_water_dimer_sapt0_ind.html",
        #                                                        )
        # Distance between monomers
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(np.sum((monA.geometry[:, None] - monB.geometry)**2, axis=2)).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r['q_A pbe0/atz']
        muA = r['mu_A pbe0/atz']
        thetaA = r['theta_A pbe0/atz']
        qB = r['q_B pbe0/atz']
        muB = r['mu_B pbe0/atz']
        thetaB = r['theta_B pbe0/atz']
        vrA = r['vol_ratios_A pbe0/atz']
        vrB = r['vol_ratios_B pbe0/atz']
        vwA = r['val_widths_A pbe0/atz']
        vwB = r['val_widths_B pbe0/atz']
        total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs = apnet_pt.multipole.eval_qcel_dimer_individual_components(
            mol_dimer=mol,
            qA=qA,
            muA=muA,
            thetaA=thetaA,
            qB=qB,
            muB=muB,
            thetaB=thetaB,
        )
        E_qq = E_qqs.sum()
        E_qu = E_qus.sum()
        E_uu = E_uus.sum()
        E_uQ = E_uQs.sum()
        E_qQ = E_qQs.sum()
        print(f"{total_energy=:.6f} kcal/mol")
        print(f"{E_qq=:.6f} kcal/mol")
        print(f"{E_qu=:.6f} kcal/mol")
        print(f"{E_uu=:.6f} kcal/mol")
        print(f"{E_qQ=:.6f} kcal/mol")
        print(f"{E_uQ=:.6f} kcal/mol")
        induction_energy = apnet_pt.multipole.dimer_induced_dipole(
            mol,
            qA=qA,
            muA=muA,
            thetaA=thetaA,
            qB=qB,
            muB=muB,
            thetaB=thetaB,
            hirshfeld_volume_ratio_A=vrA,
            hirshfeld_volume_ratio_B=vrB,
            valence_widths_A=vwA,
            valence_widths_B=vwB,
        )
        h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")
        print(f"Distance between monomers: {dist * bohr2angstrom:.2f} A")
        print(f"SAPT elst        = {sapt0_elst * h2kcalmol:.6f} kcal/mol")
        print(f"SAPT induction   = {sapt0_ind * h2kcalmol:.6f} kcal/mol")
        print(f"Induction energy = {induction_energy:.6f} kcal/mol")
        # assert abs(induction_energy - sapt0_ind) < 1e-6, f"Expected {sapt0_ind}, got {induction_energy}"


def test_classical_cliff():
    df = pd.read_pickle(file_dir + os.sep + os.path.join("dataset_data", "water_dimer_pes2.pkl"))
    pprint(df.columns.to_list())
    ap_elst, ap_ind = [], []
    ap_elst_q, ap_elst_q_mu, ap_elst_mu = [], [], []
    r = df.iloc[0]
    for n, r in df.iterrows():
        sapt0_ind = r['SAPT0 IND ENERGY adz']
        sapt0_elst = r['SAPT0 ELST ENERGY adz']
        mol = r['qcel_molecule']
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(np.sum((monA.geometry[:, None] - monB.geometry)**2, axis=2)).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r['q_A pbe0/atz']
        muA = r['mu_A pbe0/atz']
        thetaA = r['theta_A pbe0/atz']
        qB = r['q_B pbe0/atz']
        muB = r['mu_B pbe0/atz']
        thetaB = r['theta_B pbe0/atz']
        vrA = r['vol_ratios_A pbe0/atz']
        vrB = r['vol_ratios_B pbe0/atz']
        vwA = r['val_widths_A pbe0/atz']
        vwB = r['val_widths_B pbe0/atz']
        print(f"{qA = }")
        print(f"{muA = }")
        print(f"{thetaA = }")
        total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs = apnet_pt.multipole.eval_qcel_dimer_individual_components(
            mol_dimer=mol,
            qA=qA,
            muA=muA,
            thetaA=thetaA,
            qB=qB,
            muB=muB,
            thetaB=thetaB,
        )
        ap_elst.append(total_energy)
        E_qq = E_qqs.sum()
        E_qu = E_qus.sum()
        E_uu = E_uus.sum()
        ap_elst_q.append(E_qq)
        ap_elst_q_mu.append(E_qq + E_qu + E_uu)
        ap_elst_mu.append(E_uu)
        # induction_energy = apnet_pt.multipole.dimer_induced_dipole(
        #     mol,
        #     qA=qA,
        #     muA=muA,
        #     thetaA=thetaA,
        #     qB=qB,
        #     muB=muB,
        #     thetaB=thetaB,
        #     hirshfeld_volume_ratio_A=vrA,
        #     hirshfeld_volume_ratio_B=vrB,
        #     valence_widths_A=vwA,
        #     valence_widths_B=vwB,
        # )
        # ap_ind.append(induction_energy)
        # break
    print(apnet_pt.multipole.charge_dipole_qpoles_to_compact_multipoles(
        charges=qA, dipoles=muA, qpoles=thetaA,
    ))
    print(ap_elst)
    # return
    df['ap_elst'] = ap_elst
    df['ap_elst_q'] = ap_elst_q
    df['ap_elst_q_mu'] = ap_elst_q_mu
    df['ap_elst_mu'] = ap_elst_mu
    print(df[['cliff_elst', "ap_elst", 'SAPT0 ELST ENERGY adz']])
    print(df[['cliff_elst_no_nZ', "ap_elst", 'SAPT0 ELST ENERGY adz']])
    print(df[['cliff_elst_no_nZ_q', "ap_elst_q", 'SAPT0 ELST ENERGY adz']])
    print(df[['cliff_elst_no_nZ_q_mu', "ap_elst_q_mu"]])
    print(df[['cliff_elst_no_nZ_mu', "ap_elst_mu"]])
    return
    df['ap_indu'] = ap_ind
    print(df[['cliff_indu', "ap_indu", 'SAPT0 IND ENERGY adz' ]])
    print(df[['cliff_exch', 'SAPT0 EXCH ENERGY adz']])
    print(df[['cliff_disp', 'SAPT0 DISP ENERGY adz']])

# [[-8.71797290e-01 -2.01425234e-01  1.27414508e-01 -2.30556614e-15
#   -4.86321801e+00  1.04982804e-01 -7.68538422e-15  1.04982804e-01
#   -4.76251248e+00 -3.12505992e-15 -7.68538422e-15 -3.12505992e-15
#   -5.14598235e+00]
    return

if __name__ == "__main__":
    # test_induced_dipole()
    # test_induced_dipole_bz_meoh()
    test_classical_cliff()
