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
    return


if __name__ == "__main__":
    test_ap3_fused_fsapt()
