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


def test_ap3_fused_fsapt_training():
    """Test training AP3 fused model on FSAPT fragment energy data"""
    df = pd.read_pickle(f"{current_file_path}/dataset_data/fsapt_data.pkl")

    df = df.head(5)
    temp_dir = tempfile.mkdtemp()
    test_df_path = f"{data_path}/raw/fsapt_test_data.pkl"
    train_df_path = f"{data_path}/raw/fsapt_train_data.pkl"
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
            ds_spec_type=5,  # NOTE spec_type 5 for FSAPT
            ds_type="fsapt_energies",  # Important: set ds_type for FSAPT
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
    test_ap3_fused_fsapt_training()
