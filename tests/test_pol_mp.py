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


def test_train_ap3_atom_model():
    return


if __name__ == "__main__":
    test_train_ap3_atom_model()
