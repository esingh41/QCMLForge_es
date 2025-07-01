from apnet_pt import AtomPairwiseModels
import torch
import os
import pandas as pd
import apnet_pt
from glob import glob
import qcelemental as qcel


mol_dimer = qcel.models.Molecule.from_data("""
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
""")


def set_weights_to_value(model, value=0.9):
    """Sets all weights and biases in the model to a specific value."""
    print(f"Setting all weights and biases to {value}")
    with torch.no_grad():  # Disable gradient tracking
        for param in model.parameters():
            param.fill_(value)  # Set all elements to the given value


def train_pairwise_model(
    apnet_model_type="APNet2",
    model_out="./models/ap2_ensemble/ap2_1.pt",
    am_model_path="./models/ap2_ensemble/am_1.pt",
    data_dir="./data_dir",
    n_epochs=50,
    lr=5e-4,
    lr_decay=None,
    random_seed=42,
    spec_type=1,
    r_cut_im=8.0,
    r_cut=5.0,
    n_rbf=8,
    n_neuron=128,
    n_embed=8,
    m1="",
    m2="",
    pre_trained_model_path="./models/dapnet2/ap2_0.pt",
    ds_qcel_molecules=None,
    ds_energy_labels=None,
    ds_prebatched=True,
    weights=None,
):
    if ds_qcel_molecules:
        files_to_delete = glob(f"{data_dir}/processed/*None*")
        for file in files_to_delete:
            os.remove(file)

        

    ds_atomic_batch_size = 4 * 256
    ds_datapoint_storage_n_objects = 16
    if apnet_model_type == "APNet2":
        APNet = AtomPairwiseModels.apnet2.APNet2Model
    elif apnet_model_type == "APNet3":
        APNet = AtomPairwiseModels.apnet3.APNet3Model
    elif apnet_model_type == "dAPNet2":
        APNet = AtomPairwiseModels.dapnet2.dAPNet2Model
        # apnet2_model = AtomPairwiseModels.apnet2.APNet2Model().set_pretrained_model(model_id=0).model
        apnet2_model = AtomPairwiseModels.apnet2.APNet2Model(
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            r_cut_im=r_cut_im,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pre_trained_model_path,
        )
        apnet2_model.model.return_hidden_states = True
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
        print(
            apnet2_model,
            am_model_path,
            pretrained_model,
            n_rbf,
            n_neuron,
            n_embed,
            r_cut,
            r_cut_im,
            spec_type,
            data_dir,
            False,
            ds_atomic_batch_size,
            1,
            False,
            ds_datapoint_storage_n_objects,
            True,
            m1,
            m2,
        )
        apnet2 = APNet(
            apnet2_model=apnet2_model,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pretrained_model,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            r_cut_im=r_cut_im,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=ds_prebatched,
            ds_m1=m1,
            ds_m2=m2,
        )
    else:
        apnet2 = APNet(
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pretrained_model,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            r_cut_im=r_cut_im,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=ds_prebatched,
            ds_qcel_molecules=ds_qcel_molecules,
            ds_energy_labels=ds_energy_labels,
        )
    if weights is not None:
        set_weights_to_value(apnet2.atom_model, value=weights)
        set_weights_to_value(apnet2.model, value=weights)
    apnet2.train(
        model_path=model_out,
        n_epochs=n_epochs,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads_per_process,
        lr=lr,
        lr_decay=lr_decay,
        dataloader_num_workers=4,
        random_seed=random_seed,
        skip_compile=True,
    )
    return


def training_test():
    return


def main():
    am_model_path = "./models/am_ensemble/am_0.pt"
    model_out = f"./models/testing/t1"
    train_mols, train_labels = apnet_pt.util.load_dimer_dataset(
        "./tests/test_data_path/raw/t_train_100.pkl",
        columns=["Elst_aug", "Exch_aug", "Ind_aug", "Disp_aug"],
        return_qcel_mols=True,
    )
    test_mols, test_labels = apnet_pt.util.load_dimer_dataset(
        "./tests/test_data_path/raw/t_test_20.pkl",
        columns=["Elst_aug", "Exch_aug", "Ind_aug", "Disp_aug"],
        return_qcel_mols=True,
    )
    print(train_labels.shape)
    print(test_labels.shape)
    train_pairwise_model(
        apnet_model_type="APNet2",
        model_out=model_out,
        am_model_path=am_model_path,
        data_dir="./data_dir",
        n_epochs=100,
        lr=5e-4,
        lr_decay=None,
        random_seed=42,
        spec_type=None,
        r_cut_im=8.0,
        r_cut=5.0,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        ds_qcel_molecules=[train_mols, test_mols],
        ds_energy_labels=[train_labels, test_labels],
        ds_prebatched=False,
        weights=0.0,
    )
    return


if __name__ == "__main__":
    main()
