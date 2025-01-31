from apnet_pt.atom_model import AtomModel
import os
import torch

am_path = "./models/am_ensemble/am_0.pt"
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"


def test_am():
    # Test Values
    qA_ref = torch.load(f"{current_file_path}/dataset_data/mol_charges_A.pt")
    muA_ref = torch.load(f"{current_file_path}/dataset_data/mol_dipoles_A.pt")
    thetaA_ref = torch.load(f"{current_file_path}/dataset_data/mol_qpoles_A.pt")
    hlistA_ref = torch.load(f"{current_file_path}/dataset_data/mol_hlist_A.pt")
    qB_ref = torch.load(f"{current_file_path}/dataset_data/mol_charges_B.pt")
    muB_ref = torch.load(f"{current_file_path}/dataset_data/mol_dipoles_B.pt")
    thetaB_ref = torch.load(f"{current_file_path}/dataset_data/mol_qpoles_B.pt")
    hlistB_ref = torch.load(f"{current_file_path}/dataset_data/mol_hlist_B.pt")

    am = AtomModel(
        pre_trained_model_path="./models/am_ensemble/am_0.pt",
    )
    # Batch A: All full molecules
    batch_A = torch.load(f"{current_file_path}/dataset_data/batch_A.pt")
    qA, muA, thetaA, hlistA = am.predict_multipoles_batch(batch_A)
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
    batch_B = torch.load(f"{current_file_path}/dataset_data/batch_B.pt")
    qB, muB, thetaB, hlistB = am.predict_multipoles_batch(batch_B)
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
    batch_C = torch.load(f"{current_file_path}/dataset_data/batch_C.pt")
    print(batch_C)
    qC, muC, thetaC, hlistC = am.predict_multipoles_batch(batch_C)
    print("batch_C complete")
    return


if __name__ == "__main__":
    test_am()
