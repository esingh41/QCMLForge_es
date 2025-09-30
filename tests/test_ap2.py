import apnet_pt
import numpy as np
import qcelemental as qcel
import torch
import os

mol_water = qcel.models.Molecule.from_data("""
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

mol3 = qcel.models.Molecule.from_data(
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

def set_weights_to_value(model, value=0.9):
    """Sets all weights and biases in the model to a specific value."""
    with torch.no_grad():  # Disable gradient tracking
        for param in model.parameters():
            param.fill_(value)  # Set all elements to the given value


def test_ap2_architecture():
    target_energies = [
        -3.402202606201171875e01,  # ELST
        4.996978532290086150e-06,  # EXCH
        4.996978532290086150e-06,  # INDU
        4.996978532290086150e-06,  # DISP
    ]
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    set_weights_to_value(atom_model.model, 0.0001)
    pair_model = apnet_pt.AtomPairwiseModels.apnet2.APNet2Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
        use_GPU=False,
    )
    output = pair_model.predict_qcel_mols([mol3], batch_size=1)
    set_weights_to_value(pair_model.model, 0.0001)
    output = pair_model.predict_qcel_mols([mol3], batch_size=1)
    print(target_energies)
    print(output[0])
    assert np.allclose(output[0], target_energies, atol=1e-6)


def test_ap2_architecture_tf():
    target_energies = [0.06961948, 0.09670904, 0.09670904, 0.09670904]
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    set_weights_to_value(atom_model.model, 0.01)
    pair_model = apnet_pt.AtomPairwiseModels.apnet2.APNet2Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
        use_GPU=False,
    )
    output = pair_model.predict_qcel_mols([mol_water], batch_size=1)
    # NOTE: tf model logic sets atom model weights to be same, so need
    # to ensure that all are in agreement
    set_weights_to_value(pair_model.atom_model, 0.01)
    set_weights_to_value(pair_model.model, 0.01)
    output, mtp_elst = pair_model.predict_qcel_mols(
        [mol_water], batch_size=1, return_elst=True
    )
    mtp_elst = np.sum(mtp_elst)
    print(f"MTP ELST: {mtp_elst:6f}")
    print(f"NN  ELST: {(output[0][0] - mtp_elst):6f}")
    print(target_energies)
    print(output[0].tolist())
    print(f"ELST : {output[0][0]:.6f}, {target_energies[0]:.6f}")
    print(f"EXCH : {output[0][1]:.6f}, {target_energies[1]:.6f}")
    print(f"INDU : {output[0][2]:.6f}, {target_energies[2]:.6f}")
    print(f"DISP : {output[0][3]:.6f}, {target_energies[3]:.6f}")
    assert np.allclose(output[0], target_energies, atol=1e-6)



if __name__ == "__main__":
    test_ap2_architecture_tf()
    # test_ap2_predict_pairs()
