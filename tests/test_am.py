from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomModels.ap3_atom_model import AtomHirshfeldModel
import os
import torch

current_file_path = os.path.dirname(os.path.realpath(__file__))


def test_am():
    # Test Values
    qA_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_charges_A.pt", weights_only=False
    )
    muA_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_dipoles_A.pt", weights_only=False
    )
    thetaA_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_qpoles_A.pt", weights_only=False
    )
    hlistA_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_hlist_A.pt", weights_only=False
    )
    qB_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_charges_B.pt", weights_only=False
    )
    muB_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_dipoles_B.pt", weights_only=False
    )
    thetaB_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_qpoles_B.pt", weights_only=False
    )
    hlistB_ref = torch.load(
        f"{current_file_path}/dataset_data/mol_hlist_B.pt", weights_only=False
    )

    am = AtomModel(
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    # Batch A: All full molecules
    batch_A = torch.load(
        f"{current_file_path}/dataset_data/batch_A.pt", weights_only=False
    )
    qA, muA, thetaA, hlistA = am.predict_multipoles_batch(batch_A)
    # torch.save(qA, f"{current_file_path}/dataset_data/mol_charges_A.pt")
    # torch.save(muA, f"{current_file_path}/dataset_data/mol_dipoles_A.pt")
    # torch.save(thetaA, f"{current_file_path}/dataset_data/mol_qpoles_A.pt")
    # torch.save(hlistA, f"{current_file_path}/dataset_data/mol_hlist_A.pt")
    charge_cnt = 0
    for mol_charge in qA:
        charge_cnt += mol_charge.shape[0]
    assert charge_cnt == len(batch_A.x)
    for i in range(len(qA)):
        assert torch.allclose(qA[i], qA_ref[i], atol=1e-6)
        assert torch.allclose(muA[i], muA_ref[i], atol=1e-6)
        assert torch.allclose(thetaA[i], thetaA_ref[i], atol=1e-6)
        assert torch.allclose(hlistA[i], hlistA_ref[i], atol=1e-6)
    print("batch_A complete")
    # Batch B: Final molecule is single atom
    batch_B = torch.load(
        f"{current_file_path}/dataset_data/batch_B.pt", weights_only=False
    )
    qB, muB, thetaB, hlistB = am.predict_multipoles_batch(batch_B)
    # torch.save(qB, f"{current_file_path}/dataset_data/mol_charges_B.pt")
    # torch.save(muB, f"{current_file_path}/dataset_data/mol_dipoles_B.pt")
    # torch.save(thetaB, f"{current_file_path}/dataset_data/mol_qpoles_B.pt")
    # torch.save(hlistB, f"{current_file_path}/dataset_data/mol_hlist_B.pt")
    charge_cnt = 0
    for mol_charge in qB:
        charge_cnt += mol_charge.shape[0]
    print(charge_cnt, len(batch_B.x))
    assert charge_cnt == len(batch_B.x)
    for i in range(len(qB)):
        assert torch.allclose(qB[i], qB_ref[i], atol=1e-6)
        assert torch.allclose(muB[i], muB_ref[i], atol=1e-6)
        assert torch.allclose(thetaB[i], thetaB_ref[i], atol=1e-6)
        assert torch.allclose(hlistB[i], hlistB_ref[i], atol=1e-6)
    print("batch_B complete")
    batch_C = torch.load(
        f"{current_file_path}/dataset_data/batch_C.pt", weights_only=False
    )
    print(batch_C)
    qC, muC, thetaC, hlistC = am.predict_multipoles_batch(batch_C)
    print("batch_C complete")
    return


def test_am_hirshfeld():
    am = AtomHirshfeldModel(
        use_GPU=False,
        ignore_database_null=True,
    )
    return

def test_dimer_multipole_training():
    am = AtomModel(
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    return


if __name__ == "__main__":
    # test_am_hirshfeld()
    test_am()
