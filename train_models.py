from apnet_pt.atom_model import AtomModel
from apnet_pt.apnet2_model import APNet2Model
import qcelemental as qcel
from time import time
import torch
import argparse
import numpy as np
import random
import os


def train_atom_model(
    model_path="./models/am_amw_1.pt",
    spec_type=3,
    testing=False,
    data_path="data_atomic",
):
    atom_model = AtomModel(
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        ds_root=data_path,
        ds_spec_type=spec_type,
        ignore_database_null=False,
        ds_in_memory=False,
        use_GPU=True,
    )
    atom_model.train(
        n_epochs=500,
        batch_size=16,
        lr=5e-4,
        split_percent=0.9,
        model_path=model_path,
        shuffle=False,
        dataloader_num_workers=7,
        optimize_for_speed=True,
        world_size=1,
        omp_num_threads_per_process=8,
    )
    return


def train_pairwise_model(
    model_out = "./models/ap2_ensemble/ap2_1.pt",
    data_path = "./data_pairwise",
    iter=0,
):
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    print("World Size", world_size)

    batch_size = 16
    omp_num_threads_per_process = 8
    pretrained_model = model_out if os.path.exists(model_out) else None
    print(f"{pretrained_model = }")
    apnet2 = APNet2Model(
        atom_model_pre_trained_path=f"./models/am_ensemble/am_{iter}.pt",
        pre_trained_model_path=pretrained_model,
        ds_spec_type=2,
        ds_root=data_path,
        ignore_database_null=False,
        ds_atomic_batch_size=batch_size,
        ds_num_devices=1,
        ds_skip_process=False,
        ds_datapoint_storage_n_molecules=batch_size,
        ds_prebatched=True,
    )
    apnet2.train(
        model_path=model_out,
        batch_size=1,
        n_epochs=50,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads_per_process,
        lr=5e-4,
        lr_decay=None,
        dataloader_num_workers=4,
    )
    return


def set_all_seeds(seed=42, cudnn_reproducibility=False):
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # For CuDNN, setting these flags ensures reproducible but potentially
        # slower performance.
        if cudnn_reproducibility:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--am_model_output_path",
        type=str,
        default="./models/am_ensemble/am_1.pt",
        help="specify where to save output model (default: ./models/am_ensemble/am_1.pt)"
    )
    args.add_argument(
        "--ap_model_output_path",
        type=str,
        default="./models/ap2_ensemble/ap2_1.pt",
        help="specify where to save output model (default: ./models/ap2_ensemble/ap2_1.pt)"
    )
    args.add_argument(
        "--train_am",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Train AtomModel"
    )
    args.add_argument(
        "--train_ap2",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Train APNet2Model"
    )
    args.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed for initialization"
    )
    args.add_argument(
        "--spec_type",
        type=int,
        default=2,
        help="dataset spec_type recommended: (2 for AP2) or (3 for AM)"
    )
    args.add_argument(
        "--data_dir",
        type=str,
        default="./data_pairwise",
        help="specify data_dir for datasets (default: ./data_pairwise)"
    )
    args = args.parse_args()
    set_all_seeds(args.random_seed)
    if args.train_am:
        train_atom_model(args.am_model_output_path, args.data_dir)
    if args.train_ap2:
        train_pairwise_model(args.ap_model_output_path, args.data_dir, args.random_seed)
    return


if __name__ == "__main__":
    main()
