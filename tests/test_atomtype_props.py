import pytest
import apnet_pt
import qcelemental as qcel
import os
import pandas as pd
from pprint import pprint
import numpy as np
import torch

file_dir = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"


def test_elst_multipoles_MTP_torch_damping_AM_DimerParam():
    df = pd.read_pickle(
        file_dir + os.sep + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    print(r['SAPT0 ELST ENERGY adz'])
    am = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(model_id=0)
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ignore_database_null=False,
        # pre_trained_model_path="./models/am_dimer_ensemble/am_dimer_elst_damp_0.pt",
        ds_spec_type=7,
        use_GPU=False,
        ds_root=data_path,
        param_start_mean=1.5,
        param_start_std=0.1,
        n_neuron=32,
    )
    monA_props, monB_props = param_mod.predict_qcel_mols_monomer_props([mol])
    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )
    dimer_batch.qA = torch.tensor(monA_props[0][0], dtype=torch.float32)
    dimer_batch.qB = torch.tensor(monB_props[0][0], dtype=torch.float32)
    dimer_batch.muA = torch.tensor(monA_props[0][1], dtype=torch.float32)
    dimer_batch.muB = torch.tensor(monB_props[0][1], dtype=torch.float32)
    dimer_batch.quadA = torch.tensor(monA_props[0][2], dtype=torch.float32)
    dimer_batch.quadB = torch.tensor(monB_props[0][2], dtype=torch.float32)

    dimer_batch.Ka = torch.tensor(monA_props[0][4], dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(monB_props[0][4], dtype=torch.float32)

    torch_elst = apnet_pt.AtomPairwiseModels.mtp_mtp.mtp_elst_damping(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        Ka=dimer_batch.Ka,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        Kb=dimer_batch.Kb,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
    )
    print(f"Torch elst = {torch.sum(torch_elst):.6f} kcal/mol")
    return


def test_AM_hirshfeld_induction_DimerParam():
    df = pd.read_pickle(
        file_dir + os.sep + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    print(mol.to_string("psi4"))
    print(r['SAPT0 IND ENERGY adz'] * qcel.constants.hartree2kcalmol)
    am = apnet_pt.AtomModels.ap3_atom_model.AtomHirshfeldModel(
        use_GPU=False,
        ignore_database_null=True,
    )
    am.set_pretrained_model(current_file_path + "/../models/am_hf_ensemble/am_0.pt")
    # Ks = [[1.14769962, 0.685558974, 0.685558974], [1.14769962, 0.685558974, 0.685558974]]
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ignore_database_null=True,
        # pre_trained_model_path="./models/am_dimer_ensemble/am_dimer_ind_0.pt",
        ds_spec_type=7,
        use_GPU=False,
        # ds_root=data_path,
        param_start_mean=1.3,
        param_start_std=0.05,
        n_neuron=32,
        dimer_eval_type="induced_dipole",
    )
    monA_props, monB_props = param_mod.predict_qcel_mols_monomer_props([mol], am_type='ap3')
    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0, am_type='ap3',
            )
        ]
    )
    dimer_batch.qA = torch.tensor(monA_props[0][0], dtype=torch.float32)
    dimer_batch.qB = torch.tensor(monB_props[0][0], dtype=torch.float32)
    dimer_batch.muA = torch.tensor(monA_props[0][1], dtype=torch.float32)
    dimer_batch.muB = torch.tensor(monB_props[0][1], dtype=torch.float32)
    dimer_batch.quadA = torch.tensor(monA_props[0][2], dtype=torch.float32)
    dimer_batch.quadB = torch.tensor(monB_props[0][2], dtype=torch.float32)

    dimer_batch.hfvrA = torch.tensor(monA_props[0][3], dtype=torch.float32)
    dimer_batch.hfvrB = torch.tensor(monB_props[0][3], dtype=torch.float32)
    dimer_batch.vwA = torch.tensor(monA_props[0][4], dtype=torch.float32)
    dimer_batch.vwB = torch.tensor(monB_props[0][4], dtype=torch.float32)
    dimer_batch.Ka = torch.tensor(monA_props[0][-1], dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(monB_props[0][-1], dtype=torch.float32)
    # dimer_batch.Ka = torch.zeros_like(dimer_batch.Ka)
    # dimer_batch.Kb = torch.zeros_like(dimer_batch.Kb)
    print(f"Ka = {dimer_batch.Ka}")
    print(f"Kb = {dimer_batch.Kb}")

    torch_indu = apnet_pt.AtomPairwiseModels.mtp_mtp.induced_dipole_induction(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        Ka=dimer_batch.Ka,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        Kb=dimer_batch.Kb,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
        e_AA_source=dimer_batch.e_AA_source,
        e_BB_source=dimer_batch.e_BB_source,
        e_AA_target=dimer_batch.e_AA_target,
        e_BB_target=dimer_batch.e_BB_target,
        hirshfeld_volume_ratio_A=dimer_batch.hfvrA,
        hirshfeld_volume_ratio_B=dimer_batch.hfvrB,
        valence_widths_A=dimer_batch.vwA,
        valence_widths_B=dimer_batch.vwB,
    )
    print(f"Torch indu = {torch.sum(torch_indu):.6f} kcal/mol")
    pred = param_mod.predict_qcel_mols([mol, mol])
    print(pred)
    ref = np.array([[-3.38611817], [-3.38611817]])
    assert np.allclose(pred, ref, atol=1e-4)
    return


if __name__ == "__main__":
    # test_elst_multipoles_MTP_torch_damping_AM_DimerParam()
    test_AM_hirshfeld_induction_DimerParam()
