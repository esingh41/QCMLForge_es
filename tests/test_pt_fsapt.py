import apnet_pt
from apnet_pt.pt_datasets.ap3_fused_fsapt_ds import (
    ap3_fused_fsapt_module_dataset_lmdb,
    ap3_fused_fsapt_collate_update,
)
from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_Model
import os
import numpy as np
import pytest
from glob import glob
import qcelemental as qcel
import torch
import pandas as pd
from pprint import pprint as pp
import shutil
import tempfile


torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
ap3_path = f"{current_file_path}/test_models/ap3_ensemble_0/ap3_.pt"
am_hf_path = f"{current_file_path}/test_models/am_hf_0.pt"


def test_ap3_fused_fsapt():
    # Objective: train models to predict correct FSAPT pairwise energies for
    # AP3 fused model. Dataset should contain Data() with labels scalar labels
    # of F-Electrostatics, F-Exchange, F-Dispersion, F-Total, F-Induction for
    # each fragment along with correct indices to sum to match the functional
    # group breakdown. Then, the training of AP3 fused model on this type of dataset
    # will use the Frag1_indices and Frag2_indices to sum the atomic contributions
    # to get the fragment energies for computing the loss during training.

    temp_dir = tempfile.mkdtemp()
    test_df_path = f"{data_path}/raw/fsapt_test_data.pkl"
    train_df_path = f"{data_path}/raw/fsapt_train_data.pkl"
    # mkdir raw under temp_dir and copy test_df there
    raw_dir = os.path.join(temp_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Copying test dataframe to {raw_dir}")
    shutil.copy(test_df_path, raw_dir)
    shutil.copy(train_df_path, raw_dir)
    # see if the dataframe copied correctly
    print(os.listdir(raw_dir))
    test_df = pd.read_pickle(test_df_path)
    print(
        test_df[
            ["F-Electrostatics", "F-Exchange", "F-Induction", "F-Dispersion", "F-Total"]
        ].head()
    )

    # Create dataset from first 5 rows
    dataset = ap3_fused_fsapt_module_dataset_lmdb(
        root=temp_dir,
        split="test",
        spec_type=5,
        r_cut=5.0,
        r_cut_im=8.0,
        force_reprocess=True,
        random_seed=None,
    )
    print(dataset)

    try:
        # Test dataset length
        assert len(dataset) == 159, f"Expected 159 data points, got {len(dataset)}"
        print(f"Dataset length: {len(dataset)}")

        # Test getting a single data point
        data = dataset[0]
        print(f"\nFirst data point:")
        print(f"  ZA shape: {data.ZA.shape}")
        print(f"  RA shape: {data.RA.shape}")
        print(f"  ZB shape: {data.ZB.shape}")
        print(f"  RB shape: {data.RB.shape}")
        print(f"  y (FSAPT labels) shape: {data.y.shape}")
        print(f"  y values: {data.y}")
        print(f"  frag1_ind: {data.frag1_ind}")
        print(f"  frag2_ind: {data.frag2_ind}")

        # Test that labels match expected FSAPT energies
        expected_labels = torch.tensor(
            [
                test_df.iloc[0]["F-Electrostatics"],
                test_df.iloc[0]["F-Exchange"],
                test_df.iloc[0]["F-Induction"],
                test_df.iloc[0]["F-Dispersion"],
                test_df.iloc[0]["F-Total"],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(data.y, expected_labels), (
            "Labels don't match expected values"
            f"\nExpected: {expected_labels}\nGot: {data.y}"
        )
        print(f"\nLabels match expected FSAPT energies!")

        # Test batch collation
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=ap3_fused_fsapt_collate_update,
        )

        batch = next(iter(dataloader))
        print(f"\nBatched data:")
        print(f"  Batch y shape: {batch.y.shape}")
        print(f"  Batch ZA shape: {batch.ZA.shape}")
        print(f"  Batch RA shape: {batch.RA.shape}")
        print(f"  Number of fragment 1 indices: {len(batch.frag1_ind)}")
        print(f"  Number of fragment 2 indices: {len(batch.frag2_ind)}")

        # Verify batch size
        assert batch.y.shape[0] == 2, f"Expected batch size 2, got {batch.y.shape[0]}"
        assert len(batch.frag1_ind) == 2, "Expected 2 fragment 1 index tensors"
        assert len(batch.frag2_ind) == 2, "Expected 2 fragment 2 index tensors"

        print("\nAll tests passed!")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def test_ap3_fused_fsapt_energies():
    """Test training AP3 fused model on FSAPT fragment energy data"""
    df = pd.read_pickle(f"{data_path}/raw/fsapt_train_simple.pkl")
    print(df)
    print(df.columns.tolist())
    fAs = {}
    fBs = {}
    for f1, inds in zip(df['Frag1'], df['Frag1_indices']):
        fAs[f1] = inds
    for f2, inds in zip(df['Frag2'], df['Frag2_indices']):
        fBs[f2] = inds
    print(df.iloc[0]['qcel_molecule'].to_string('psi4'))
    pred_IEs, pairwise_energies, df = apnet_pt.pretrained_models.apnet3_model_predict_pairs(
        df['qcel_molecule'].tolist(),
        fAs=[fAs for i in range(len(df))],
        fBs=[fBs for i in range(len(df))],
    )
    print(df)
    """
                  fA-fB      total       elst      exch      indu      disp
0         Methyl1_A-All  12.485888  12.711604  0.110707 -0.017730 -0.318692
1   Methyl1_A-Peptide_B  12.721631  12.720556  0.002140  0.007264 -0.008330
2   Methyl1_A-T-Butyl_B  -0.235743  -0.008952  0.108566 -0.024994 -0.310362
3         Methyl2_A-All  11.814778  10.555523  4.227958 -0.381347 -2.587355
4   Methyl2_A-Peptide_B  10.450938  10.490627  0.005944  0.011028 -0.056661
5   Methyl2_A-T-Butyl_B   1.363840   0.064896  4.222014 -0.392375 -2.530694
6               All-All  24.300666  23.267127  4.338664 -0.399078 -2.906048
7         All-Peptide_B  23.172568  23.211183  0.008084  0.018292 -0.064991
8         All-T-Butyl_B   1.128098   0.055943  4.330580 -0.417369 -2.841057
9         Methyl1_A-All -17.293892 -17.018704  0.110707 -0.067202 -0.318692
10  Methyl1_A-Peptide_B -17.058149 -17.009752  0.002140 -0.042208 -0.008330
11  Methyl1_A-T-Butyl_B  -0.235743  -0.008952  0.108566 -0.024994 -0.310362
12        Methyl2_A-All  -1.723804  -2.968268  4.227958 -0.396138 -2.587355
13  Methyl2_A-Peptide_B  -3.087644  -3.033164  0.005944 -0.003763 -0.056661
14  Methyl2_A-T-Butyl_B   1.363840   0.064896  4.222014 -0.392375 -2.530694
15              All-All -19.017696 -19.986972  4.338664 -0.463340 -2.906048
16        All-Peptide_B -20.145794 -20.042916  0.008084 -0.045971 -0.064991
17        All-T-Butyl_B   1.128098   0.055943  4.330580 -0.417369 -2.841057
18        Methyl1_A-All -13.955007 -13.723142  0.110707 -0.023880 -0.318692
19  Methyl1_A-Peptide_B -13.719264 -13.714189  0.002140  0.001115 -0.008330
20  Methyl1_A-T-Butyl_B  -0.235743  -0.008952  0.108566 -0.024994 -0.310362
21        Methyl2_A-All   4.098228   2.850917  4.227958 -0.393291 -2.587355
22  Methyl2_A-Peptide_B   2.734388   2.786021  0.005944 -0.000916 -0.056661
23  Methyl2_A-T-Butyl_B   1.363840   0.064896  4.222014 -0.392375 -2.530694
24              All-All  -9.856779 -10.872225  4.338664 -0.417171 -2.906048
25        All-Peptide_B -10.984876 -10.928168  0.008084  0.000199 -0.064991
26        All-T-Butyl_B   1.128098   0.055943  4.330580 -0.417369 -2.841057
27        Methyl1_A-All   5.024506   5.265885  0.110707 -0.033394 -0.318692
28  Methyl1_A-Peptide_B   5.260248   5.274838  0.002140 -0.008400 -0.008330
29  Methyl1_A-T-Butyl_B  -0.235743  -0.008952  0.108566 -0.024994 -0.310362
30        Methyl2_A-All   2.373583   1.126893  4.227958 -0.393913 -2.587355
31  Methyl2_A-Peptide_B   1.009742   1.061997  0.005944 -0.001538 -0.056661
32  Methyl2_A-T-Butyl_B   1.363840   0.064896  4.222014 -0.392375 -2.530694
33              All-All   7.398088   6.392779  4.338664 -0.427307 -2.906048
34        All-Peptide_B   6.269991   6.336835  0.008084 -0.009938 -0.064991
35        All-T-Butyl_B   1.128098   0.055943  4.330580 -0.417369 -2.841057
36        Methyl1_A-All   9.336811   9.547227  0.110707 -0.002430 -0.318692
37  Methyl1_A-Peptide_B   9.572554   9.556179  0.002140  0.022565 -0.008330
38  Methyl1_A-T-Butyl_B  -0.235743  -0.008952  0.108566 -0.024994 -0.310362
39        Methyl2_A-All  15.129613  13.813276  4.227958 -0.324266 -2.587355
40  Methyl2_A-Peptide_B  13.765773  13.748380  0.005944  0.068110 -0.056661
41  Methyl2_A-T-Butyl_B   1.363840   0.064896  4.222014 -0.392375 -2.530694
42              All-All  24.466424  23.360503  4.338664 -0.326695 -2.906048
43        All-Peptide_B  23.338326  23.304559  0.008084  0.090674 -0.064991
44        All-T-Butyl_B   1.128098   0.055943  4.330580 -0.417369 -2.841057
    """
    return

def test_ap3_fused_fsapt_training():
    """Test training AP3 fused model on FSAPT fragment energy data"""
    temp_dir = tempfile.mkdtemp()
    test_df_path = f"{data_path}/raw/fsapt_test_simple.pkl"
    train_df_path = f"{data_path}/raw/fsapt_train_simple.pkl"
    # mkdir raw under temp_dir and copy test_df there
    raw_dir = os.path.join(temp_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Copying test dataframe to {raw_dir}")
    shutil.copy(test_df_path, raw_dir)
    shutil.copy(train_df_path, raw_dir)
    try:
        # Initialize atom models (required for AP3)
        atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_path,
            pre_trained_model_path=at_hf_vw_path,
        )
        atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=at_elst_path,
        )

        # Initialize AP3 model for FSAPT training

        ap3 = APNet3_AtomType_Model(
            # ds_root=temp_dir,
            ds_root=data_path,
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            am_dimer_param_model=atom_type_elst_model,
            use_precomputed_classical=False,
            ignore_database_null=False,
            ds_spec_type=6,  # NOTE spec_type 5 for FSAPT
            ds_type="fsapt_energies",  # Important: set ds_type for FSAPT
            pre_trained_model_path=ap3_path,
        )

        print("\nStarting FSAPT training...")

        # Train for a few epochs
        ap3.train(
            n_epochs=0,
            lr=5e-5,
            skip_compile=True,  # Skip compilation for faster testing
        )

        print("\nFSAPT training test completed successfully!")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    return


if __name__ == "__main__":
    # test_ap3_fused_fsapt()
    # test_ap3_fused_fsapt_training()
    test_ap3_fused_fsapt_energies()
