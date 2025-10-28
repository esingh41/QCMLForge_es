import apnet_pt
from apnet_pt import AtomPairwiseModels
from apnet_pt import pairwise_datasets
from apnet_pt.pairwise_datasets import (
    apnet2_module_dataset,
    apnet2_collate_update,
    apnet2_collate_update_prebatched,
    APNet2_DataLoader,
    apnet3_module_dataset,
    apnet3_collate_update,
    apnet3_collate_update_prebatched,
)
import os
import numpy as np
import pytest
from glob import glob
import qcelemental as qcel
import torch
import pandas as pd
from pprint import pprint as pp


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
    df = pd.read_pickle(f"{current_file_path}/dataset_data/fsapt_data.pkl")
    print(df)
    pp(df.columns.tolist())
    # Prints
    """
     Frag1   Frag2                                      Frag1_indices  ...    F-Total                                      qcel_molecule  F-Induction
0   ILE462     All  [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...  ...  -1.520982  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.081967
1   GLY463     All                                   [20, 22, 23, 24]  ...  -0.116115  Molecule(name='C48ClH73N12O9S', formula='C48Cl...     0.000913
2   ALA480     All                               [25, 26, 27, 28, 29]  ...  -0.914920  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.142219
3   THR528     All               [30, 31, 32, 33, 34, 35, 36, 37, 38]  ...   0.347610  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.043630
4   GLN529     All                                   [39, 42, 43, 44]  ...  -0.207636  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.075938
5   TRP530     All  [46, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 6...  ...  -4.679316  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.470981
6   CYS531     All                           [70, 73, 76, 77, 78, 79]  ...   0.399682  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.112764
7   GLU532     All                                       [81, 85, 86]  ...  -0.277294  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.038290
8   GLY533     All                                       [88, 92, 93]  ...   0.858525  Molecule(name='C48ClH73N12O9S', formula='C48Cl...     0.158538
9   SER534     All                   [95, 98, 99, 101, 102, 103, 104]  ...  -0.966036  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.084600
10  SER535     All                               [106, 108, 109, 110]  ...   0.110007  Molecule(name='C48ClH73N12O9S', formula='C48Cl...     0.042755
11  PHE582     All  [111, 112, 113, 114, 115, 116, 117, 118, 119, ...  ...  -5.992308  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.326716
12   DIS74     All                                               [74]  ...  -1.549153  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.129065
13  PEP462     All                                     [2, 3, 19, 21]  ...  -0.092489  Molecule(name='C48ClH73N12O9S', formula='C48Cl...     0.049837
14  PEP529     All                                   [40, 41, 45, 59]  ...  -1.903547  Molecule(name='C48ClH73N12O9S', formula='C48Cl...     0.107949
15  PEP530     All                                   [47, 48, 69, 75]  ...  -5.088104  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -2.281762
16  PEP531     All                                   [71, 72, 80, 84]  ...  -9.052702  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -2.582748
17  PEP532     All                                   [82, 83, 87, 91]  ...   1.581134  Molecule(name='C48ClH73N12O9S', formula='C48Cl...     0.292450
18  PEP533     All                                  [89, 90, 94, 100]  ...  -8.291030  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -1.574728
19  PEP534     All                                 [96, 97, 105, 107]  ...   0.092955  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -0.092548
20     All  LIGAND  [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...  ... -37.261721  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -7.385512
21     All     All  [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...  ... -37.261721  Molecule(name='C48ClH73N12O9S', formula='C48Cl...    -7.385512

[22 rows x 11 columns]
['Frag1',
 'Frag2',
 'Frag1_indices',
 'Frag2_indices',
 'F-Electrostatics',
 'F-Exchange',
 'F-Dispersion',
 'F-EDispersion',
 'F-Total',
 'qcel_molecule',
 'F-Induction']
    """
    # Objective: train models to predict correct FSAPT pairwise energies for
    # AP3 fused model. Dataset should contain Data() with labels scalar labels
    # of F-Electrostatics, F-Exchange, F-Dispersion, F-Total, F-Induction for
    # each fragment along with correct indices to sum to match the functional
    # group breakdown. Then, the training of AP3 fused model on this type of dataset
    # will use the Frag1_indices and Frag2_indices to sum the atomic contributions
    # to get the fragment energies for computing the loss during training.
    
    # Test the new AP3FusedFSAPTDataset
    from apnet_pt.pt_datasets.ap3_fused_fsapt_ds import (
        AP3FusedFSAPTDataset,
        ap3_fused_fsapt_collate_update,
        fsapt_dimer_to_fused_data,
    )
    
    # Create dataset from first 5 rows
    test_df = df.head(5)
    dataset = AP3FusedFSAPTDataset(
        root=data_path,
        fsapt_dataframe=test_df,
        r_cut=5.0,
        r_cut_im=8.0,
        force_reprocess=True,
    )
    
    # Test dataset length
    assert len(dataset) == 5, f"Expected 5 data points, got {len(dataset)}"
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
    print(f"  frag1_indices: {data.frag1_indices}")
    print(f"  frag2_indices: {data.frag2_indices}")
    print(f"  frag1_name: {data.frag1_name}")
    print(f"  frag2_name: {data.frag2_name}")
    
    # Test that labels match expected FSAPT energies
    expected_labels = torch.tensor([
        test_df.iloc[0]['F-Electrostatics'],
        test_df.iloc[0]['F-Exchange'],
        test_df.iloc[0]['F-Dispersion'],
        test_df.iloc[0]['F-Induction'],
        test_df.iloc[0]['F-Total'],
    ], dtype=torch.float32)
    assert torch.allclose(data.y, expected_labels), "Labels don't match expected values"
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
    print(f"  Number of fragment 1 indices: {len(batch.frag1_indices)}")
    print(f"  Number of fragment 2 indices: {len(batch.frag2_indices)}")
    
    # Verify batch size
    assert batch.y.shape[0] == 2, f"Expected batch size 2, got {batch.y.shape[0]}"
    assert len(batch.frag1_indices) == 2, "Expected 2 fragment 1 index tensors"
    assert len(batch.frag2_indices) == 2, "Expected 2 fragment 2 index tensors"
    
    print("\nAll tests passed!")
    return


def test_ap3_fused_fsapt_with_multipoles():
    """Test AP3FusedFSAPTDataset with multipole prediction"""
    df = pd.read_pickle(f"{current_file_path}/dataset_data/fsapt_data.pkl")
    
    # Load atom model for multipole prediction
    from apnet_pt.AtomModels.ap2_atom_model import AtomModel
    atom_model = AtomModel(use_GPU=False).set_pretrained_model(model_id=0)
    
    # Test the new AP3FusedFSAPTDataset with multipoles
    from apnet_pt.pt_datasets.ap3_fused_fsapt_ds import (
        AP3FusedFSAPTDataset,
        ap3_fused_fsapt_collate_update,
    )
    
    # Create dataset from first 3 rows with multipoles
    test_df = df.head(3)
    dataset = AP3FusedFSAPTDataset(
        root=data_path,
        fsapt_dataframe=test_df,
        r_cut=5.0,
        r_cut_im=8.0,
        force_reprocess=True,
        atom_model=atom_model,
    )
    
    # Test dataset length
    assert len(dataset) == 3, f"Expected 3 data points, got {len(dataset)}"
    print(f"Dataset with multipoles length: {len(dataset)}")
    
    # Test getting a single data point
    data = dataset[0]
    print(f"\nFirst data point with multipoles:")
    print(f"  ZA shape: {data.ZA.shape}")
    print(f"  y (FSAPT labels) shape: {data.y.shape}")
    
    # Check multipoles are present
    assert hasattr(data, 'qA'), "Missing qA multipole"
    assert hasattr(data, 'muA'), "Missing muA multipole"
    assert hasattr(data, 'quadA'), "Missing quadA multipole"
    assert hasattr(data, 'hlistA'), "Missing hlistA multipole"
    assert hasattr(data, 'qB'), "Missing qB multipole"
    assert hasattr(data, 'muB'), "Missing muB multipole"
    assert hasattr(data, 'quadB'), "Missing quadB multipole"
    assert hasattr(data, 'hlistB'), "Missing hlistB multipole"
    
    print(f"  qA shape: {data.qA.shape}")
    print(f"  muA shape: {data.muA.shape}")
    print(f"  quadA shape: {data.quadA.shape}")
    print(f"  hlistA shape: {data.hlistA.shape}")
    print(f"  qB shape: {data.qB.shape}")
    print(f"  muB shape: {data.muB.shape}")
    print(f"  quadB shape: {data.quadB.shape}")
    print(f"  hlistB shape: {data.hlistB.shape}")
    
    # Test batch collation with multipoles
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=ap3_fused_fsapt_collate_update,
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatched data with multipoles:")
    print(f"  Batch y shape: {batch.y.shape}")
    print(f"  Batch qA shape: {batch.qA.shape}")
    print(f"  Batch muA shape: {batch.muA.shape}")
    print(f"  Batch quadA shape: {batch.quadA.shape}")
    print(f"  Batch hlistA shape: {batch.hlistA.shape}")
    
    # Verify multipoles are batched correctly
    assert hasattr(batch, 'qA'), "Missing qA in batched data"
    assert hasattr(batch, 'muA'), "Missing muA in batched data"
    assert hasattr(batch, 'qB'), "Missing qB in batched data"
    assert hasattr(batch, 'muB'), "Missing muB in batched data"
    
    print("\nMultipole tests passed!")
    return


def test_ap3_fused_fsapt_lmdb():
    """Test AP3FusedFSAPTDatasetLMDB"""
    df = pd.read_pickle(f"{current_file_path}/dataset_data/fsapt_data.pkl")
    
    # Test the LMDB dataset
    from apnet_pt.pt_datasets.ap3_fused_fsapt_ds import (
        AP3FusedFSAPTDatasetLMDB,
        ap3_fused_fsapt_collate_update,
    )
    
    # Create LMDB dataset from first 5 rows
    test_df = df.head(5)
    dataset = AP3FusedFSAPTDatasetLMDB(
        root=data_path,
        fsapt_dataframe=test_df,
        r_cut=5.0,
        r_cut_im=8.0,
        force_reprocess=True,
        cache_size=10,
    )
    
    # Test dataset length
    assert len(dataset) == 5, f"Expected 5 data points, got {len(dataset)}"
    print(f"LMDB dataset length: {len(dataset)}")
    
    # Test getting a single data point
    data = dataset[0]
    print(f"\nFirst data point from LMDB:")
    print(f"  ZA shape: {data.ZA.shape}")
    print(f"  RA shape: {data.RA.shape}")
    print(f"  y (FSAPT labels) shape: {data.y.shape}")
    print(f"  frag1_indices: {data.frag1_indices}")
    
    # Test random access
    data_3 = dataset[3]
    print(f"\nData point 3 from LMDB:")
    print(f"  y values: {data_3.y}")
    
    # Test batch loading from LMDB
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=ap3_fused_fsapt_collate_update,
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatched data from LMDB:")
    print(f"  Batch y shape: {batch.y.shape}")
    print(f"  Batch ZA shape: {batch.ZA.shape}")
    
    # Verify batch size
    assert batch.y.shape[0] == 2, f"Expected batch size 2, got {batch.y.shape[0]}"
    
    print("\nLMDB tests passed!")
    return


if __name__ == "__main__":
    test_ap3_fused_fsapt()
    test_ap3_fused_fsapt_with_multipoles()
    test_ap3_fused_fsapt_lmdb()
