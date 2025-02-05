import apnet_pt
import qcelemental
import torch
import os
import numpy as np

# mol_mon = qcelemental.models.Molecule.from_data("""1 1
# C       0.0545060001    -0.1631290019   -1.1141539812
# C       -0.9692260027   -1.0918780565   0.6940879822
# C       0.3839910030    0.5769280195    -0.0021170001
# C       1.3586950302    1.7358809710    0.0758149996
# N       -0.1661809981   -0.0093130004   1.0584640503
# N       -0.8175240159   -1.0993789434   -0.7090409994
# H       0.3965460062    -0.1201139987   -2.1653149128
# H       -1.5147459507   -1.6961929798   1.3000769615
# H       0.7564010024    2.6179349422    0.4376020133
# H       2.2080008984    1.5715960264    0.7005280256
# H       1.7567750216    2.0432629585    -0.9004560113
# H       -0.1571149975   0.2784340084    1.9974440336
# H       -1.2523859739   -1.9090379477   -1.2904200554
# units angstrom
# """)
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

mol_dimer = qcelemental.models.Molecule.from_data("""1 1
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
    return


def test_ap2_ensemble():
    # ref = torch.load(os.path.join(os.path.dirname(
    #     __file__), "dataset_data/ap2_ensemble_test.pt"))
    #
    mols = [mol_dimer for _ in range(3)]
    interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
        mols,
        compile=False,
        batch_size=2,
    )
    torch.save(interaction_energies, os.path.join(os.path.dirname(
        __file__), "dataset_data/ap2_ensemble_test.pt"))
    return


if __name__ == "__main__":
    # test_am_ensemble()
    test_ap2_ensemble()
