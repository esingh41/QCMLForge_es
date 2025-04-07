from apnet_pt import AtomModels
from apnet_pt import AtomPairwiseModels
import torch
import argparse
import numpy as np
import random
import os
from pprint import pprint


def train_atom_model(
    atom_model_type="AtomModel",
    model_path="./models/am_amw_1.pt",
    data_dir="data_atomic",
    spec_type=3,
    testing=False,
    n_epochs=500,
    random_seed=42,
    ds_max_size=None,
):
    if atom_model_type == "AtomModel":
        AM = AtomModels.ap2_atom_model.AtomModel
        batch_size = 16
    elif atom_model_type == "AtomHirshfeldModel":
        AM = AtomModels.ap3_atom_model.AtomHirshfeldModel
        batch_size = 1
    else:
        raise ValueError("Invalid Atom Model Type")
    pretrained_model = None
    if os.path.exists(model_path):
        pretrained_model = model_path
    print("Training {}...".format(atom_model_type))
    atom_model = AM(
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        ds_root=data_dir,
        ds_spec_type=spec_type,
        ds_max_size=ds_max_size,
        ignore_database_null=False,
        ds_in_memory=True,
        use_GPU=True,
        pre_trained_model_path=pretrained_model,
    )
    print(atom_model.dataset)
    atom_model.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=5e-4,
        split_percent=0.9,
        model_path=model_path,
        shuffle=False,
        dataloader_num_workers=7,
        optimize_for_speed=True,
        world_size=1,
        omp_num_threads_per_process=8,
        random_seed=random_seed,
    )
    return


def train_pairwise_model(
    apnet_model_type="APNet2",
    model_out="./models/ap2_ensemble/ap2_1.pt",
    am_model_path="./models/ap2_ensemble/am_1.pt",
    data_dir="./data_pairwise",
    n_epochs=50,
    lr=5e-4,
    lr_decay=None,
    random_seed=42,
    spec_type=2,
    m1="",
    m2="",
):
    ds_atomic_batch_size = 4 * 256
    ds_datapoint_storage_n_objects = 16
    if apnet_model_type == "APNet2":
        APNet = AtomPairwiseModels.apnet2.APNet2Model
    elif apnet_model_type == "APNet3":
        APNet = AtomPairwiseModels.apnet3.APNet3Model
    elif apnet_model_type == "dAPNet2":
        APNet = AtomPairwiseModels.dapnet2.dAPNet2Model
        apnet2_model = AtomPairwiseModels.apnet2.APNet2Model().set_pretrained_model(model_id=0).model
        apnet2_model.return_hidden_states = True
    else:
        raise ValueError("Invalid Atom Model Type")
    print("Training {}...".format(apnet_model_type))
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    print("World Size", world_size)

    omp_num_threads_per_process = 8
    if os.path.exists(model_out):
        pretrained_model = model_out
        print(f"\nTraining from {model_out}\n")
    else:
        pretrained_model = None
        print("\nTraining from scratch...\n")
    if apnet_model_type.startswith("dAPNet"):
        apnet2 = APNet(
            apnet2_model=apnet2_model,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pretrained_model,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=True,
            ds_m1=m1,
            ds_m2=m2,
        )
    else:
        apnet2 = APNet(
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pretrained_model,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=True,
        )
    apnet2.train(
        model_path=model_out,
        n_epochs=n_epochs,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads_per_process,
        lr=lr,
        lr_decay=lr_decay,
        dataloader_num_workers=1,
        random_seed=random_seed,
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
        type=str,
        default="",
        help="Train AtomModel: (AtomModel, AtomHirshfeldModel)"
    )
    args.add_argument(
        "--train_apnet",
        type=str,
        default="",
        help="Train APNet Model: (APNet2, APNet3, dAPNet2)"
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
        default=3,
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
        "--ds_max_size",
        type=int,
        default=None,
        help="Limit dataset to N dataset objects",
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
    args.add_argument(
        "--m1",
        type=str,
        default="",
        help="specify dAP-Net level of theory 1 (default: '')"
    )
    args.add_argument(
        "--m2",
        type=str,
        default="",
        help="specify dAP-Net level of theory 2 (default: '')"
    )
    args = args.parse_args()
    pprint(args)
    set_all_seeds(args.random_seed)
    if args.train_am != "":
        train_atom_model(
            atom_model_type=args.train_am,
            model_path=args.am_model_path,
            data_dir=args.data_dir_atom,
            spec_type=args.spec_type_am,
            n_epochs=args.n_epochs_atom,
            random_seed=args.random_seed,
            ds_max_size=args.ds_max_size,
        )
    if args.train_apnet != "":
        train_pairwise_model(
            apnet_model_type=args.train_apnet,
            model_out=args.ap_model_path,
            am_model_path=args.am_model_path,
            data_dir=args.data_dir,
            n_epochs=args.n_epochs,
            lr=args.lr,
            lr_decay=args.lr_decay,
            random_seed=args.random_seed,
            spec_type=args.spec_type_ap,
            m1=args.m1,
            m2=args.m2,
        )
    return


if __name__ == "__main__":
    main()
