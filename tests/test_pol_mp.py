import apnet_pt
from apnet_pt import atomic_datasets
from apnet_pt import AtomModels
import os
import numpy as np
import pytest
from glob import glob
import qcelemental as qcel
import torch
import pandas as pd
from pprint import pprint as pp

# Handle both pytest and direct execution
try:
    from . import mols
except ImportError:
    import mols


torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
atp_path = f"{current_file_path}/test_models/ap3_ensemble_0/atp_mpnn_1.pt"
# aidm_path = f"{current_file_path}/test_models/ap3_ensemble_0/atomInducedDipole_atp_screeningNN_1.pt"
aidm_path = f"{current_file_path}/test_models/ap3_ensemble_0/atomInducedDipole_atp_screeningNN_lr_1.pt"


def test_train_ap3_atomTypeparamMPNN():
    ds = atomic_datasets.atomic_hirshfeld_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=16,
    )
    print(ds)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    atpm.train(
        n_epochs=3,
        batch_size=16,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=4,
        random_seed=42,
    )
    return


def test_train_ap3_atom_model():
    ds = atomic_datasets.atomic_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=6,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    )
    print(ds)
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
        pre_trained_model_path=atp_path,
    )
    print(atpm)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    am.train(
        n_epochs=3,
        batch_size=1,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=4,
        random_seed=42,
    )
    return


def test_inference_ap3_atom_model():
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=aidm_path,
        use_nn_screening=True,
    )
    am.compile_model()
    v = am.predict_qcel_mols(
        mols=[
            mols.lr_water_dimer,
            mols.lr_water_dimer,
            # mols.mol_cliff_water_close,
        ],
        batch_size=2,
    )
    print(f"{v = }")
    # torch save v for reference
    # torch.save(v, f"{current_file_path}/../debug_ap3_atom_model_inference.pt")
    ref = torch.load(f"{current_file_path}/../debug_ap3_atom_model_inference.pt")
    for i in range(len(v[0])):
        assert torch.allclose(v[0][i], ref[0][i], atol=1e-6), f"{i}, {v[0][i]}, {ref[0][i]}"
    return


def train_ap3_atom_model():
    ds = atomic_datasets.atomic_module_dataset(
        root=f"{current_file_path}/../data_dimer_1",
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=10,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=16,
    )
    torch.set_printoptions(profile="full")
    print(ds)
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
        pre_trained_model_path=atp_path,
    )
    print(atpm)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    am.train(
        n_epochs=100,
        batch_size=16,
        lr=5e-4,
        split_percent=0.9,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=4,
        random_seed=42,
        model_path=f"{current_file_path}/..models/ap3_ensemble/ap3_am_spec10.pt",
    )
    return


def test_lmdb_dataset_creation():
    """Test LMDB dataset creation and basic functionality"""
    # Load pre-trained atomtype model
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Create LMDB dataset
    from apnet_pt.atomic_datasets import atomic_module_dataset_lmdb

    ds_lmdb = atomic_module_dataset_lmdb(
        root=data_path,
        atomtype_hfvr_model=atpm.model,
        testing=False,
        spec_type=5,
        max_size=50,  # Small for testing
        force_reprocess=False,
        in_memory=False,
        cache_size=10,
    )

    print(f"LMDB dataset length: {len(ds_lmdb)}")
    assert len(ds_lmdb) > 0, "Dataset should have items"

    # Test item retrieval
    item = ds_lmdb[0]
    assert hasattr(item, "x"), "Item should have node features"
    assert hasattr(item, "edge_index"), "Item should have edges"
    assert hasattr(item, "volume_ratios"), "Item should have volume ratios"
    assert hasattr(item, "valence_widths"), "Item should have valence widths"

    print("LMDB dataset tests passed!")
    return


def test_train_with_lmdb():
    """Test training AtomInducedDipoleModel with LMDB dataset"""
    # Load pre-trained atomtype model
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Set up environment
    os.environ["OMP_NUM_THREADS"] = "4"

    # Create AtomInducedDipoleModel with LMDB dataset
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        ds_root=data_path,
        ds_spec_type=5,
        ds_max_size=50,  # Small for testing
        ds_use_lmdb=True,  # KEY: Use LMDB dataset
        ds_in_memory=False,  # Don't load all into memory
        precompute_hfvr=True,  # Pre-compute hfvr/vw during dataset processing
    )

    # Train (single process)
    am.train(
        n_epochs=2,
        batch_size=2,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,  # Single process
        omp_num_threads_per_process=4,
        random_seed=42,
    )
    print("Training with LMDB dataset completed successfully!")
    return


def test_ddp_train_ap3_atom_model():
    """Test distributed data parallel training with world_size=2"""
    # Load pre-trained atomtype model
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Set up DDP environment
    os.environ["OMP_NUM_THREADS"] = "2"

    # Create AtomInducedDipoleModel with LMDB dataset
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        ds_root=data_path,
        ds_spec_type=5,
        ds_max_size=50,  # Small for testing
        ds_use_lmdb=True,  # KEY: Use LMDB dataset
        ds_in_memory=False,  # Don't load all into memory
        precompute_hfvr=True,  # Pre-compute hfvr/vw during dataset processing
    )

    # Train with DDP (world_size=2)
    am.train(
        n_epochs=2,
        batch_size=2,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=2,  # KEY: DDP with 2 processes
        omp_num_threads_per_process=2,
        random_seed=42,
    )
    return


def debug_ap3_atom_model():
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        dataset=None,
        pre_trained_model_path=atp_path,
    )
    print(atpm)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=True,
        dataset=None,
    )
    am.model(torch.load(f"{current_file_path}/../debug_batch.pt", weights_only=False))
    return


if __name__ == "__main__":
    # test_train_ap3_atomTypeparamMPNN()
    # test_train_ap3_atom_model()
    # train_ap3_atom_model()
    # debug_ap3_atom_model()
    # test_inference_ap3_atom_model()
    test_ddp_train_ap3_atom_model()
