from apnet_pt.apnet2_model import APNet2Model
from apnet_pt import atomic_datasets
from apnet_pt import AtomModels
from apnet_pt.pairwise_datasets import (
    apnet2_module_dataset,
    apnet2_collate_update,
    APNet2_DataLoader,
)
import pickle
import os
import numpy as np
import pytest

spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../models/am_ensemble/am_0.pt"
ap_path = f"{current_file_path}/../models/ap2_ensemble/ap2_0.pt"



def test_apnet_data_object():
    # TF batch
    with open(f"{current_file_path}/dataset_data/inp_batch0.pkl", "rb") as f:
        inp_batch0 = pickle.load(f)
    with open(f"{current_file_path}/dataset_data/ie_batch0.pkl", "rb") as f:
        ie_batch0 = pickle.load(f)
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        atom_model_path=am_path,
        atomic_batch_size=1000,
        num_devices=1,
        skip_processed=False,
        split="train",
    )
    batch_size = 16

    train_loader = APNet2_DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=apnet2_collate_update,
    )
    batch1 = next(iter(train_loader))
    batch1_ie = batch1.y
    print(ie_batch0)
    print(batch1_ie)
    assert np.allclose(ie_batch0, batch1_ie[:, :4])
    print(ds)
    print(batch1)
    print(inp_batch0['RA'].shape)
    for k, v in inp_batch0.items():
        if k in ["monomerA_ind", "monomerB_ind"]:
            continue
        print(k, v.shape)
        assert np.allclose(v, getattr(batch1, k).numpy())


@pytest.mark.skip(reason="Slow training test. Run only for development reasons.")
def test_apnet2_model_train():
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        atom_model_path=am_path,
        atomic_batch_size=1000,
        num_devices=1,
        skip_processed=False,
        split="train",
    )
    apnet2 = APNet2Model(
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=ap_path,
        dataset=ds,
        ds_root=data_path,
        ds_spec_type=spec_type,
        ds_force_reprocess=False,
        ignore_database_null=False,
        ds_atomic_batch_size=1000,
        ds_num_devices=1,
        ds_skip_process=False,
        # ds_max_size=10,
    )
    apnet2.train(
        model_path="./models/ap2_test.pt",
        batch_size=16,
        n_epochs=1,
        world_size=1,
        omp_num_threads_per_process=8,
        lr=2e-3,
        lr_decay=0.10,
        # lr_decay=None,
    )
    return

def test_atomhirshfeld_model_train():
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
        batch_size=1,
    )
    print(ds)
    am = AtomModels.ap3_atom_model.AtomHirshfeldModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    print(am)
    am.train(
        n_epochs=5,
        batch_size=1,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        optimize_for_speed=False,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=None,
        random_seed=42,
    )
    return


if __name__ == "__main__":
    # test_apnet_data_object()
    # test_apnet2_model_train()
    test_atomhirshfeld_model_train()
