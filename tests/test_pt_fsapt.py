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
    return


if __name__ == "__main__":
    test_ap3_fused_fsapt()
