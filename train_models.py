from apnet_pt.atom_model import AtomModel
from apnet_pt.apnet2_model import APNet2Model
import torch
import argparse
import numpy as np
import random
import os
from pprint import pprint


def train_atom_model(
    model_path="./models/am_amw_1.pt",
    data_dir="data_atomic",
    spec_type=3,
    testing=False,
    n_epochs=500,
):
    atom_model = AtomModel(
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        ds_root=data_dir,
        ds_spec_type=spec_type,
        ignore_database_null=False,
        ds_in_memory=False,
        use_GPU=True,
    )
    atom_model.train(
        n_epochs=n_epochs,
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
    model_out="./models/ap2_ensemble/ap2_1.pt",
    am_model_path="./models/ap2_ensemble/am_1.pt",
    data_dir="./data_pairwise",
    n_epochs=50,
    lr=5e-4,
    lr_decay=None,
):
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    print("World Size", world_size)

    batch_size = 16
    omp_num_threads_per_process = 8
    if os.path.exists(model_out):
        pretrained_model = model_out
        print(f"\nTraining from {model_out}\n")
    else:
        pretrained_model = None
        print("\nTraining from scratch...\n")
    apnet2 = APNet2Model(
        atom_model_pre_trained_path=am_model_path,
        pre_trained_model_path=pretrained_model,
        ds_spec_type=2,
        ds_root=data_dir,
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
        n_epochs=n_epochs,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads_per_process,
        # lr=5e-4,
        # lr=2e-3,
        # lr_decay=0.10,
        lr=lr,
        lr_decay=lr_decay,
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
        "--am_model_path",
        type=str,
        default="./models/am_ensemble/am_0.pt",
        help="specify where to save output model (default: ./models/am_ensemble/am_1.pt)"
    )
    args.add_argument(
        "--ap_model_path",
        type=str,
        default="./models/ap2_ensemble/ap2_0.pt",
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
        default=0,
        help="Random seed for initialization"
    )
    args.add_argument(
        "--spec_type_am",
        type=int,
        default=2,
        help="dataset spec_type recommended: (3 for AM)"
    )
    args.add_argument(
        "--spec_type_ap",
        type=int,
        default=2,
        help="dataset spec_type recommended: (2 for AP2)"
    )
    args.add_argument(
        "--data_dir_atom",
        type=str,
        default="./data_dir",
        help="specify data_dir for datasets (default: ./data_dir)"
    )
    args.add_argument(
        "--data_dir",
        type=str,
        default="./data_dir",
        help="specify data_dir for datasets (default: ./data_dir)"
    )
    args.add_argument(
        "--n_epochs_atom",
        type=int,
        default=500,
        help="Number of epochs for training"
    )
    args.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Number of epochs for training"
    )
    args.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning Rate: (5e-4 is default)"
    )
    args.add_argument(
        "--lr_decay",
        type=float,
        default=None,
        help="Learning Rate Decay: (None is default, takes in float)"
    )
    args = args.parse_args()
    pprint(args)
    set_all_seeds(args.random_seed)
    if args.train_am:
        train_atom_model(
            model_path=args.am_model_path,
            data_dir=args.data_dir_atomic,
            spec_type=args.spec_type_am,
            n_epochs=args.n_epochs_atom,
            random_seed=args.random_seed,
        )
    if args.train_ap2:
        train_pairwise_model(
            model_out=args.ap_model_path,
            am_model_path=args.am_model_path,
            data_dir=args.data_dir,
            n_epochs=args.n_epochs,
            lr=args.lr,
            lr_decay=args.lr_decay,
            random_seed=args.random_seed,
        )
    return


if __name__ == "__main__":
    main()
