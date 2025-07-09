from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomModels.ap3_atom_model import AtomHirshfeldModel
import os
import torch
import apnet_pt
import qcelemental as qcel
import numpy as np

current_file_path = os.path.dirname(os.path.realpath(__file__))

mol_water = qcel.models.Molecule.from_data("""
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
""")


def set_weights_to_value(model, value=0.9):
    """Sets all weights and biases in the model to a specific value."""
    with torch.no_grad():  # Disable gradient tracking
        for param in model.parameters():
            param.fill_(value)  # Set all elements to the given value


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


def allclose_sigfig(a, b, sigfigs=4, equal_nan=False):
    """
    Returns True if all elements in arrays `a` and `b` agree to the specified number of significant figures.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    sigfigs : int
        Number of significant figures required to match.
    equal_nan : bool
        Whether to consider NaNs equal.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return False

    # Handle NaNs if requested
    if equal_nan:
        nan_mask = np.isnan(a) & np.isnan(b)
    else:
        nan_mask = np.zeros_like(a, dtype=bool)

    # To avoid division by zero or log10(0), mask out zeros temporarily
    with np.errstate(divide="ignore", invalid="ignore"):
        # Compute the order of magnitude of each value
        magnitude = np.floor(np.log10(np.maximum(np.abs(a), np.abs(b))))
        scale = 10 ** (magnitude - sigfigs + 1)
        diff = np.abs(a - b)

    close = (diff <= scale) | nan_mask
    return np.all(close)


def test_am_architecture():
    tf_charges = np.array([0.28356934, -0.14175415, -0.14175415], dtype=np.float32)
    tf_dipoles = np.array(
        [
            [17.078526, 0.06, -6.0958734],
            [-17.774773, 0.06, -22.643368],
            [0.8762475, 0.06, 28.919243],
        ],
        dtype=np.float32,
    )
    tf_quads = np.array(
        [
            [
                1.9083790e03,
                5.9999999e-02,
                -1.1078560e03,
                -1.1545651e03,
                5.9999999e-02,
                -7.5381415e02,
            ],
            [
                4.2554663e02,
                5.9999999e-02,
                4.2820825e03,
                -2.9382349e03,
                5.9999999e-02,
                2.5126885e03,
            ],
            [
                -2.9311895e03,
                5.9999999e-02,
                2.4915816e02,
                -2.9382344e03,
                5.9999999e-02,
                5.8694233e03,
            ],
        ],
        dtype=np.float32,
    )


    hidden_list_v = [np.array([[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
           [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
           [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]], dtype=np.float32),
    np.array([[1.0521429, 1.0521429, 1.0521429, 1.0521429, 1.0521429, 1.0521429,
            1.0521429, 1.0521429],
           [1.0513331, 1.0513331, 1.0513331, 1.0513331, 1.0513331, 1.0513331,
            1.0513331, 1.0513331],
           [1.0513331, 1.0513331, 1.0513331, 1.0513331, 1.0513331, 1.0513331,
            1.0513331, 1.0513331]], dtype=np.float32),
    np.array([[13.901527, 13.901527, 13.901527, 13.901527, 13.901527, 13.901527,
            13.901527, 13.901527],
           [13.890074, 13.890074, 13.890074, 13.890074, 13.890074, 13.890074,
            13.890074, 13.890074],
           [13.890074, 13.890074, 13.890074, 13.890074, 13.890074, 13.890074,
            13.890074, 13.890074]], dtype=np.float32),
    np.array([[173.86403, 173.86403, 173.86403, 173.86403, 173.86403, 173.86403,
            173.86403, 173.86403],
           [173.7179 , 173.7179 , 173.7179 , 173.7179 , 173.7179 , 173.7179 ,
            173.7179 , 173.7179 ],
           [173.7179 , 173.7179 , 173.7179 , 173.7179 , 173.7179 , 173.7179 ,
            173.7179 , 173.7179 ]], dtype=np.float32)]

    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    set_weights_to_value(atom_model.model, 0.02)
    v = atom_model.predict_qcel_mols([mol_water], batch_size=1)
    charges, dipoles, quads, hlist = v[0]
    charges = charges.detach().cpu().numpy()
    dipoles = dipoles.detach().cpu().numpy()
    hlist = hlist.detach().cpu().numpy()
    quads = quads.detach().cpu().numpy()
    quads = quads.reshape(-1, 9)
    quads = np.array(
        [quads[i].flatten()[[0, 1, 2, 4, 5, 8]] for i in range(len(quads))]
    )
    print(f"{charges=}\n{dipoles=}\n{quads=}")
    hlist = np.transpose(hlist, (1, 0, 2))
    assert allclose_sigfig(charges, tf_charges, sigfigs=3)
    assert allclose_sigfig(dipoles, tf_dipoles, sigfigs=3)
    assert allclose_sigfig(quads, tf_quads, sigfigs=3)
    assert allclose_sigfig(hlist, hidden_list_v, sigfigs=3), f"{hlist - hidden_list_v=}"


if __name__ == "__main__":
    # test_am_hirshfeld()
    # test_am()
    test_am_architecture()
