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
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    am = AtomModels.ap3_atom_model.AtomModel(
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


if __name__ == "__main__":
    test_train_ap3_atom_model()
