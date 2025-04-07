import torch
import os
from .. import util
from ..pairwise_datasets import (
    AtomModel,
    Data,
    Dataset,
    pairwise_edges,
    pairwise_edges_im,
    natural_key,
    apnet2_collate_update,
)
from .. import atomic_datasets
from time import time
from glob import glob
from pathlib import Path
import os.path as osp

# file dir
prefix = os.path.dirname(os.path.abspath(__file__))
pretrained_atom_model_path = prefix + "/../models/am_ensemble/"

def clean_str_for_filename(string):
    """
    Remove all non-alphanumeric characters from a string and replace spaces with underscores.
    """
    # Remove all non-alphanumeric characters
    string = string.replace("(", "_LP_").replace(")", "_RP_")
    string = ''.join(e for e in string if e.isalnum() or e.isspace())
    # Replace spaces with underscores
    string = string.replace(' ', '_')
    return string

class dapnet2_module_dataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=1,
        max_size=None,
        force_reprocess=True,
        skip_processed=True,
        # only need for processing
        atom_model_path=pretrained_atom_model_path + "am_0.pt",
        batch_size=16,
        atomic_batch_size=256,
        prebatched=False,
        # DO NOT CHANGE UNLESS YOU WANT TO RE-PROCESS THE DATASET
        datapoint_storage_n_objects=1000,
        in_memory=False,
        num_devices=1,
        split="all",  # train, test
        print_level=2,
        m1="B3LYP-D3/aug-cc-pVTZ/CP",
        m2="CCSD(T)/CBS/CP",
    ):
        """
        spec_type definitions:
            1. dapnet dataset for bfdb interaction energies
        """
        self.print_level = print_level
        try:
            assert spec_type in [1, 8]
        except Exception:
            print("Currently spec_type must be 1 or 2 for SAPT0/jun-cc-pVDZ")
            raise ValueError
        self.spec_type = spec_type
        self.prebatched = prebatched
        self.MAX_SIZE = max_size
        self.m1 = m1
        self.m2 = m2
        self.split = split
        self.r_cut = r_cut
        self.r_cut_im = r_cut_im
        self.force_reprocess = force_reprocess
        self.filename_methods = clean_str_for_filename(m1) + "_to_" + clean_str_for_filename(m2)
        self.datapoint_storage_n_molecules = datapoint_storage_n_objects
        self.atomic_batch_size = atomic_batch_size
        self.batch_size = batch_size
        self.training_batch_size = batch_size if not prebatched else 1
        self.datapoint_storage_n_objects = datapoint_storage_n_objects
        self.points_per_file = self.datapoint_storage_n_objects
        if self.prebatched:
            self.points_per_file *= self.batch_size
        self.in_memory = in_memory
        self.skip_processed = skip_processed
        if os.path.exists(root) is False:
            os.makedirs(root, exist_ok=True)
        if atom_model_path is not None and not self.skip_processed:
            self.atom_model = AtomModel(
                pre_trained_model_path=atom_model_path,
                ds_root=None,
                ignore_database_null=True,
            )
            self.atom_model.model.to(self.atom_model.device)
            torch._dynamo.config.dynamic_shapes = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True
            if False:
                self.atom_model.model = torch.compile(
                    self.atom_model.model, dynamic=True)
        super(dapnet2_module_dataset, self).__init__(
            root, transform, pre_transform)
        print(
            f"{self.root=}, {self.spec_type=}, {self.in_memory=}"
        )
        if self.in_memory:
            print("Loading data into memory")
            t = time()
            self.data = []
            for i in self.processed_file_names:
                self.data.append(torch.load(os.path.join(self.processed_dir, i), weights_only=False))
            total_time_seconds = int(time() - t)
            print(f"Loaded in {total_time_seconds:4d} seconds")
            self.get = self.get_in_memory
        # self.active_data = [None for i in self.processed_file_names]
        self.active_idx_data = None
        self.active_data = None

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed_delta')

    @property
    def raw_file_names(self):
        # TODO: enable users to specify data source via QCArchive, url, or local file
        # spec_1 = "spec_1" # 'SAPT0/jun-cc-pVDZ'
        if self.spec_type == 1:
            return [
                # 0.4 Train / 0.6 Test
                "3324_BFDBext_train_dimers.pkl",
                "3324_BFDBext_test_dimers.pkl",
            ]
        elif self.spec_type == 8:
            return [
                "t_val_19.pkl",
            ]
        else:
            return [
                "splinter_spec1.pkl",
            ]

    def reprocess_file_names(self):
        if self.force_reprocess:
            return ["file"]
        else:
            if self.split == "train":
                file_cmd = f"{self.processed_dir}/dimer_dap2_train_spec_{self.spec_type}_{self.filename_methods}*.pt"
            elif self.split == "test":
                file_cmd = f"{self.processed_dir}/dimer_dap2_test_spec_{self.spec_type}_{self.filename_methods}*.pt"
            else:
                file_cmd = f"{self.processed_dir}/dimer_dap2_spec_{self.spec_type}_{self.filename_methods}*.pt"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                spec_files.sort(key=natural_key)
                if self.MAX_SIZE is not None:
                    max_size = int(self.MAX_SIZE /
                                   self.datapoint_storage_n_molecules)
                if self.MAX_SIZE is not None:
                    if len(spec_files) > max_size and max_size > 0:
                        spec_files = spec_files[:max_size]
                    elif len(spec_files) > max_size:
                        spec_files = spec_files[:1]
                return spec_files
            else:
                return ["dimer_missing.pt"]

    @property
    def processed_file_names(self):
        return self.reprocess_file_names()

    def download(self):
        raise RuntimeError(
            "Dataset does not have a download method. Development usage only at the moment."
        )
        return

    def process(self):
        idx = 0
        batch_size = self.batch_size
        atomic_batch_size = self.atomic_batch_size
        if self.prebatched:
            assert atomic_batch_size % batch_size == 0, f"atomic_batch_size {atomic_batch_size} must be divisible by batch_size {batch_size}"
        if self.spec_type in [1, 2, 7]:
            print(f"ENSURE THAT {atomic_batch_size=} is the same as the batch size used in the AP-Net2 model training! This mode avoids collating completely.")
        data_objects = []
        for raw_path in self.raw_paths:
            split_name = ""
            if self.spec_type in [1]:
                idx = 0
                split_name = f"_{self.split}"
                print(f"{split_name=}")
                if self.split not in Path(raw_path).stem:
                    print(f"{self.split} is skipping {raw_path}")
                    continue
            print(f"raw_path: {raw_path}")
            print("Loading dimers...")
            RAs, RBs, ZAs, ZBs, TQAs, TQBs, targets = util.load_dimer_dataset(
                raw_path, self.MAX_SIZE, return_qcel_mols=False, return_qcel_mons=False,
                columns=[self.m1, self.m2],
            )
            print("Creating data objects...")
            t1 = time()
            t2 = time()
            print(f"{len(RAs)=}, {atomic_batch_size=}")
            molA_data = []
            molB_data = []
            energies = []
            for i in range(0, len(RAs) + len(RAs) % self.atomic_batch_size + 1, self.atomic_batch_size):
                if self.skip_processed:
                    datapath = osp.join(
                        self.processed_dir,
                        f"dimer_dap2{split_name}_spec_{self.spec_type}_{self.filename_methods}_{idx // self.points_per_file}.pt",
                    )
                    if osp.exists(datapath):
                        idx += atomic_batch_size
                        continue
                upper_bound = min(i + atomic_batch_size, len(RAs))
                for j in range(i, upper_bound):
                    monA_data = atomic_datasets.create_atomic_data(
                        ZAs[j], RAs[j], TQAs[j], r_cut=self.r_cut
                    )
                    monB_data = atomic_datasets.create_atomic_data(
                        ZBs[j], RBs[j], TQBs[j], r_cut=self.r_cut
                    )
                    molA_data.append(monA_data)
                    molB_data.append(monB_data)
                    energies.append(targets[j])
                if len(molA_data) != self.atomic_batch_size and j != len(RAs) - 1:
                    continue
                batch_A = atomic_datasets.atomic_collate_update_no_target(molA_data)
                qAs, muAs, thAs, hlistAs = self.atom_model.predict_multipoles_batch(batch_A)
                batch_B = atomic_datasets.atomic_collate_update_no_target(molB_data)
                qBs, muBs, thBs, hlistBs = self.atom_model.predict_multipoles_batch(batch_B)
                for j in range(len(molA_data)):
                    atomic_props_A = molA_data[j]
                    atomic_props_B = molB_data[j]
                    local_energies = energies[j]
                    qA, muA, quadA, hlistA = qAs[j], muAs[j], thAs[j], hlistAs[j]
                    qB, muB, quadB, hlistB = qBs[j], muBs[j], thBs[j], hlistBs[j]
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                    e_AA_source, e_AA_target = pairwise_edges(atomic_props_A.R, self.r_cut)
                    e_BB_source, e_BB_target = pairwise_edges(atomic_props_B.R, self.r_cut)
                    e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = pairwise_edges_im(
                        atomic_props_A.R, atomic_props_B.R, self.r_cut_im
                    )
                    y = torch.tensor(local_energies, dtype=torch.float32)
                    dimer_ind = torch.ones((1), dtype=torch.long) * j
                    data = Data(
                        y=y,
                        ZA=atomic_props_A.x,
                        RA=atomic_props_A.R,
                        ZB=atomic_props_B.x,
                        RB=atomic_props_B.R,
                        e_ABsr_source=e_ABsr_source,
                        e_ABsr_target=e_ABsr_target,
                        dimer_ind=dimer_ind,
                        e_ABlr_source=e_ABlr_source,
                        e_ABlr_target=e_ABlr_target,
                        dimer_ind_lr=dimer_ind,
                        e_AA_source=e_AA_source,
                        e_AA_target=e_AA_target,
                        e_BB_source=e_BB_source,
                        e_BB_target=e_BB_target,
                        total_charge_A=atomic_props_A.total_charge,
                        total_charge_B=atomic_props_B.total_charge,
                        qA=qA,
                        muA=muA,
                        quadA=quadA,
                        hlistA=hlistA,
                        qB=qB,
                        muB=muB,
                        quadB=quadB,
                        hlistB=hlistB,
                    )
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    data_objects.append(data)

                    if (
                        len(data_objects) == self.points_per_file
                    ):
                        datapath = osp.join(
                            self.processed_dir,
                            f"dimer_dap2{split_name}_spec_{self.spec_type}_{self.filename_methods}_{idx // self.points_per_file}.pt",
                        )
                        if self.print_level >= 2:
                            print(f"Saving to {datapath}")
                            print(len(data_objects))
                        # if we are pre-batching, we need to collate and save here.
                        if self.prebatched:
                            # collate based on batch_size
                            local_data_objects = []
                            for k in range(self.datapoint_storage_n_objects):
                                local_data_objects.append(apnet2_collate_update(data_objects[k * self.batch_size:(k + 1) * self.batch_size]))
                            data_objects = local_data_objects
                        torch.save(data_objects, datapath)
                        data_objects = []
                        if self.MAX_SIZE is not None and idx > self.MAX_SIZE:
                            break
                    idx += 1
                if self.print_level >= 2:
                    print(f"{i}/{len(RAs)}, {time()-t2:.2f}s, {time()-t1:.2f}s")
                elif self.print_level >= 1 and idx % 1000:
                    print(f"{i}/{len(RAs)}, {time()-t2:.2f}s, {time()-t1:.2f}s")
                t2 = time()
                molA_data = []
                molB_data = []
                energies = []
        if len(data_objects) > 0:
            if self.prebatched:
                # collate based on batch_size
                local_data_objects = []
                for k in range(len(data_objects) // self.batch_size):
                    local_data_objects.append(apnet2_collate_update(data_objects[k * self.batch_size:(k + 1) * self.batch_size]))
                data_objects = local_data_objects
            datapath = osp.join(
                self.processed_dir,
                    f"dimer_dap2{split_name}_spec_{self.spec_type}_{self.filename_methods}_{idx // self.points_per_file}.pt",
            )
            if self.print_level >= 2:
                print(f"Final Saving to {datapath}")
                print(len(data_objects))
            torch.save(data_objects, datapath)
        return

    def len(self):
        d = torch.load(
            osp.join(self.processed_dir, self.processed_file_names[-1]), weights_only=False
        )
        if self.prebatched:
            return (len(self.processed_file_names) - 1) * self.datapoint_storage_n_objects + len(d)
        return (len(self.processed_file_names) - 1) * self.datapoint_storage_n_objects + len(d)

    def get(self, idx):
        idx_datapath = idx // self.datapoint_storage_n_molecules
        obj_ind = idx % self.datapoint_storage_n_molecules
        if self.active_idx_data == idx_datapath:
            return self.active_data[obj_ind]
        split_name = ""
        if self.spec_type in [2]:
            split_name = f"_{self.split}"
        datapath = os.path.join(
            self.processed_dir, f"dimer_dap2{split_name}_spec_{
                self.spec_type}_{self.filename_methods}_{idx_datapath}.pt"
        )
        if self.spec_type in [1]:
            return torch.load(datapath, weights_only=False)
        self.active_data = torch.load(datapath, weights_only=False)
        return self.active_data[obj_ind]
