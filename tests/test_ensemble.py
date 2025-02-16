import apnet_pt
import qcelemental
import torch
import os
import numpy as np

mol_mon = qcelemental.models.Molecule.from_data("""0 1
16  -0.8795  -2.0832  -0.5531
7   -0.2959  -1.8177   1.0312
7    0.5447  -0.7201   1.0401
6    0.7089  -0.1380  -0.1269
6    0.0093  -0.7249  -1.1722
1    1.3541   0.7291  -0.1989
1   -0.0341  -0.4523  -2.2196
units angstrom
""")

mol_dimer = qcelemental.models.Molecule.from_data("""
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
""")


def test_am_ensemble():
    print("Testing AM ensemble...")
    ref = torch.load(os.path.join(os.path.dirname(
        __file__), "dataset_data/am_ensemble_test.pt"))

    mols = [mol_mon for _ in range(3)]
    multipoles = apnet_pt.pretrained_models.atom_model_predict(
        mols,
        compile=False,
        batch_size=2,
    )
    q_ref = ref[0]
    q = multipoles[0]
    assert np.allclose(q, q_ref, atol=1e-6)
    d_ref = ref[1]
    d = multipoles[1]
    assert np.allclose(d, d_ref, atol=1e-6)
    qp_ref = ref[2]
    qp = multipoles[2]
    assert np.allclose(qp, qp_ref, atol=1e-6)


def test_ap2_ensemble():
    print("Testing AP2 ensemble...")
    ref = torch.load(os.path.join(os.path.dirname(
        __file__), "dataset_data/ap2_ensemble_test.pt"))

    mols = [mol_dimer for _ in range(3)]
    interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
        mols,
        compile=False,
        batch_size=2,
    )
    assert np.allclose(interaction_energies, ref, atol=1e-6)


def test_am_ensemble_compile():
    print("Testing AM ensemble...")
    ref = torch.load(os.path.join(os.path.dirname(
        __file__), "dataset_data/am_ensemble_test.pt"))

    mols = [mol_mon for _ in range(3)]
    multipoles = apnet_pt.pretrained_models.atom_model_predict(
        mols,
        compile=True,
        batch_size=2,
    )
    q_ref = ref[0]
    q = multipoles[0]
    assert np.allclose(q, q_ref, atol=1e-6)
    d_ref = ref[1]
    d = multipoles[1]
    assert np.allclose(d, d_ref, atol=1e-6)
    qp_ref = ref[2]
    qp = multipoles[2]
    assert np.allclose(qp, qp_ref, atol=1e-6)


def test_ap2_ensemble_compile():
    print("Testing AP2 ensemble...")
    ref = torch.load(os.path.join(os.path.dirname(
        __file__), "dataset_data/ap2_ensemble_test.pt"))

    mols = [mol_dimer for _ in range(3)]
    interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
        mols,
        compile=True,
        batch_size=2,
    )
    assert np.allclose(interaction_energies, ref, atol=1e-6)


if __name__ == "__main__":
    test_am_ensemble()
    # test_ap2_ensemble()
