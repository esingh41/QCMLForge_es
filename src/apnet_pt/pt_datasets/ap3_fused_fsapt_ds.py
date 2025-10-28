"""
AP3-fused FSAPT dataset for training models on fragment-based SAPT energies.

This dataset stores FSAPT breakdown energies (F-Electrostatics, F-Exchange, 
F-Dispersion, F-Induction, F-Total) along with fragment indices (Frag1_indices, 
Frag2_indices) to enable training AP3 fused models on fragment-level energies.

During training, the model predicts atomic-level contributions which are summed
according to the fragment indices to compute fragment-level energies for loss 
calculation against the FSAPT reference values.
"""

import os
import os.path as osp
import qcelemental as qcel
from typing import List, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import pandas as pd
from pathlib import Path
from apnet_pt import constants

from .ap3_fused_ds import (
    dimer_fused_data,
    pairwise_edges,
    pairwise_edges_im,
    natural_key,
)


def fsapt_dimer_to_fused_data(
    row,
    r_cut=5.0,
    r_cut_im=8.0,
    dimer_ind=0,
    check_validity=True,
):
    """
    Convert FSAPT dataframe row to AP3 fused data object with fragment labels.
    
    Parameters
    ----------
    row : pd.Series
        Row from FSAPT dataframe containing:
        - qcel_molecule: QCElemental molecule object (dimer)
        - Frag1_indices: List of atom indices for fragment 1
        - Frag2_indices: List of atom indices for fragment 2
        - F-Electrostatics: FSAPT electrostatics energy
        - F-Exchange: FSAPT exchange energy
        - F-Dispersion: FSAPT dispersion energy
        - F-Induction: FSAPT induction energy
        - F-Total: Total FSAPT energy
    r_cut : float
        Cutoff radius for intra-monomer edges (Angstrom)
    r_cut_im : float
        Cutoff radius for inter-monomer edges (Angstrom)
    dimer_ind : int
        Index of dimer in dataset
    check_validity : bool
        Check if molecular featurization is valid
        
    Returns
    -------
    Data
        PyTorch Geometric Data object with monomer geometries, edges, and FSAPT labels
    """
    dimer = row['qcel_molecule']
    
    # Extract monomers
    monA = dimer.get_fragment(0)
    monB = dimer.get_fragment(1)
    
    # Get coordinates and atomic numbers
    RA = torch.tensor(monA.geometry, dtype=torch.float32) * constants.au2ang
    RB = torch.tensor(monB.geometry, dtype=torch.float32) * constants.au2ang
    ZA = torch.tensor(monA.atomic_numbers, dtype=torch.int64)
    ZB = torch.tensor(monB.atomic_numbers, dtype=torch.int64)
    TQA = torch.tensor(monA.molecular_charge, dtype=torch.float32)
    TQB = torch.tensor(monB.molecular_charge, dtype=torch.float32)
    
    # Extract FSAPT energies as labels
    fsapt_labels = torch.tensor([
        row['F-Electrostatics'],
        row['F-Exchange'],
        row['F-Dispersion'],
        row['F-Induction'],
        row['F-Total'],
    ], dtype=torch.float32)
    
    # Extract fragment indices
    frag1_indices = torch.tensor(row['Frag1_indices'], dtype=torch.long)
    frag2_indices = torch.tensor(row['Frag2_indices'], dtype=torch.long)
    
    # Create base dimer data
    data = dimer_fused_data(
        RA=RA,
        ZA=ZA,
        TQA=TQA,
        RB=RB,
        ZB=ZB,
        TQB=TQB,
        dimer_ind=dimer_ind,
        r_cut=r_cut,
        r_cut_im=r_cut_im,
        check_validity=check_validity,
        y=fsapt_labels,
        frag1_indices=frag1_indices,
        frag2_indices=frag2_indices,
        frag1_name=row['Frag1'],
        frag2_name=row['Frag2'],
    )
    
    return data


def ap3_fused_fsapt_collate_update(batch):
    """
    Collate function for AP3 fused FSAPT dataset.
    
    Similar to ap3_fused_collate_update but also handles fragment indices
    for computing fragment-level energies during training.
    
    Parameters
    ----------
    batch : List[Data]
        List of Data objects from dataset
        
    Returns
    -------
    Data
        Batched Data object with all necessary tensors and fragment indices
    """
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []
    local_e_ABfull_source = []
    local_e_ABfull_target = []
    
    local_e_AA_source = []
    local_e_AA_target = []
    local_e_BB_source = []
    local_e_BB_target = []
    
    # Fragment index tracking
    local_frag1_indices = []
    local_frag2_indices = []
    atom_offset_A = 0
    atom_offset_B = 0
    
    for i, data in enumerate(batch):
        data.dimer_ind = (
            torch.ones(data.e_ABsr_source.size(0), dtype=data.dimer_ind.dtype) * i
        )
        data.dimer_ind_lr = (
            torch.ones(data.e_ABlr_source.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
        data.dimer_ind_full = (
            torch.ones(
                data.e_ABlr_source.size(0) + data.e_ABsr_source.size(0),
                dtype=data.dimer_ind_lr.dtype
            ) * i
        )
        
        data.molecule_ind_A = (
            torch.ones(data.ZA.size(0), dtype=torch.long) * i
        )
        data.molecule_ind_B = (
            torch.ones(data.ZB.size(0), dtype=torch.long) * i
        )
        
        local_e_ABsr_source.append(data.e_ABsr_source.clone() + monA_edge_offset)
        local_e_ABsr_target.append(data.e_ABsr_target.clone() + monB_edge_offset)
        local_e_ABlr_source.append(data.e_ABlr_source.clone() + monA_edge_offset)
        local_e_ABlr_target.append(data.e_ABlr_target.clone() + monB_edge_offset)
        
        local_e_ABfull_source.append(
            torch.cat([
                data.e_ABsr_source.clone() + monA_edge_offset,
                data.e_ABlr_source.clone() + monA_edge_offset
            ])
        )
        local_e_ABfull_target.append(
            torch.cat([
                data.e_ABsr_target.clone() + monB_edge_offset,
                data.e_ABlr_target.clone() + monB_edge_offset
            ])
        )
        
        local_e_AA_source.append(data.e_AA_source.clone() + monA_edge_offset)
        local_e_AA_target.append(data.e_AA_target.clone() + monA_edge_offset)
        local_e_BB_source.append(data.e_BB_source.clone() + monB_edge_offset)
        local_e_BB_target.append(data.e_BB_target.clone() + monB_edge_offset)
        
        # Offset fragment indices for batching
        if hasattr(data, 'frag1_indices') and data.frag1_indices is not None:
            local_frag1_indices.append(data.frag1_indices.clone() + atom_offset_A)
        if hasattr(data, 'frag2_indices') and data.frag2_indices is not None:
            local_frag2_indices.append(data.frag2_indices.clone() + atom_offset_B)
        
        monA_edge_offset += data.RA.size(0)
        monB_edge_offset += data.RB.size(0)
        atom_offset_A += data.ZA.size(0)
        atom_offset_B += data.ZB.size(0)
    
    molecule_ind_A = torch.cat([data.molecule_ind_A for data in batch], dim=0)
    molecule_ind_B = torch.cat([data.molecule_ind_B for data in batch], dim=0)
    natom_per_mol_A = torch.bincount(molecule_ind_A)
    natom_per_mol_B = torch.bincount(molecule_ind_B)
    y = torch.stack([data.y for data in batch], dim=0)
    
    ZA_cat = torch.cat([data.ZA for data in batch], dim=0)
    RA_cat = torch.cat([data.RA for data in batch], dim=0)
    ZB_cat = torch.cat([data.ZB for data in batch], dim=0)
    RB_cat = torch.cat([data.RB for data in batch], dim=0)
    e_AA_source_cat = torch.cat(local_e_AA_source, dim=0)
    e_AA_target_cat = torch.cat(local_e_AA_target, dim=0)
    e_BB_source_cat = torch.cat(local_e_BB_source, dim=0)
    e_BB_target_cat = torch.cat(local_e_BB_target, dim=0)
    total_charge_A_tensor = torch.tensor(
        [data.total_charge_A for data in batch], dtype=batch[0].total_charge_A.dtype
    )
    total_charge_B_tensor = torch.tensor(
        [data.total_charge_B for data in batch], dtype=batch[0].total_charge_B.dtype
    )
    
    batch_atomic_A = Data(
        x=ZA_cat,
        edge_index=torch.vstack((e_AA_source_cat, e_AA_target_cat)),
        R=RA_cat,
        molecule_ind=molecule_ind_A,
        total_charge=total_charge_A_tensor,
        natom_per_mol=natom_per_mol_A,
    )
    
    batch_atomic_B = Data(
        x=ZB_cat,
        edge_index=torch.vstack((e_BB_source_cat, e_BB_target_cat)),
        R=RB_cat,
        molecule_ind=molecule_ind_B,
        total_charge=total_charge_B_tensor,
        natom_per_mol=natom_per_mol_B,
    )
    
    e_ABsr_source_cat = torch.cat(local_e_ABsr_source, dim=0)
    e_ABsr_target_cat = torch.cat(local_e_ABsr_target, dim=0)
    e_ABlr_source_cat = torch.cat(local_e_ABlr_source, dim=0)
    e_ABlr_target_cat = torch.cat(local_e_ABlr_target, dim=0)
    
    e_ABfull_source = torch.cat(local_e_ABfull_source, dim=0)
    e_ABfull_target = torch.cat(local_e_ABfull_target, dim=0)
    
    dimer_ind_cat = torch.cat([data.dimer_ind for data in batch], dim=0)
    dimer_ind_lr_cat = torch.cat([data.dimer_ind_lr for data in batch], dim=0)
    dimer_ind_full_cat = torch.cat([data.dimer_ind_full for data in batch], dim=0)
    
    batched_data = Data(
        y=y,
        ZA=ZA_cat,
        RA=RA_cat,
        ZB=ZB_cat,
        RB=RB_cat,
        e_AA_source=e_AA_source_cat,
        e_AA_target=e_AA_target_cat,
        e_BB_source=e_BB_source_cat,
        e_BB_target=e_BB_target_cat,
        molecule_ind_A=molecule_ind_A,
        molecule_ind_B=molecule_ind_B,
        natom_per_mol_A=natom_per_mol_A,
        natom_per_mol_B=natom_per_mol_B,
        e_ABsr_source=e_ABsr_source_cat,
        e_ABsr_target=e_ABsr_target_cat,
        e_ABlr_source=e_ABlr_source_cat,
        e_ABlr_target=e_ABlr_target_cat,
        e_ABfull_source=e_ABfull_source,
        e_ABfull_target=e_ABfull_target,
        dimer_ind=dimer_ind_cat,
        dimer_ind_lr=dimer_ind_lr_cat,
        dimer_ind_full=dimer_ind_full_cat,
        total_charge_A=total_charge_A_tensor,
        total_charge_B=total_charge_B_tensor,
        batch_atomic_A=batch_atomic_A,
        batch_atomic_B=batch_atomic_B,
    )
    
    # Add fragment indices if present
    if len(local_frag1_indices) > 0:
        batched_data.frag1_indices = local_frag1_indices
        batched_data.frag1_batch_ind = torch.tensor(
            [i for i in range(len(batch))], dtype=torch.long
        )
    if len(local_frag2_indices) > 0:
        batched_data.frag2_indices = local_frag2_indices
        batched_data.frag2_batch_ind = torch.tensor(
            [i for i in range(len(batch))], dtype=torch.long
        )
    
    return batched_data


class AP3FusedFSAPTDataset(Dataset):
    """
    Dataset for training AP3 fused models on FSAPT fragment energies.
    
    This dataset loads FSAPT data from pandas DataFrames containing:
    - qcel_molecule: QCElemental Molecule objects (dimers)
    - Frag1_indices, Frag2_indices: Atom indices defining fragments
    - F-Electrostatics, F-Exchange, F-Dispersion, F-Induction, F-Total: FSAPT energies
    
    The dataset creates PyTorch Geometric Data objects with:
    - Monomer geometries and atomic numbers (ZA, RA, ZB, RB)
    - Intra- and inter-molecular edges
    - Fragment indices for each data point
    - FSAPT energy labels for training
    
    During training, models predict atomic-level energies which are summed
    according to fragment indices to compute fragment-level predictions for
    comparison with FSAPT reference values.
    
    Parameters
    ----------
    root : str
        Root directory for storing processed data
    fsapt_dataframe : pd.DataFrame, optional
        DataFrame with FSAPT data (if not provided, will try to load from processed files)
    r_cut : float
        Cutoff radius for intra-monomer edges (Angstrom)
    r_cut_im : float
        Cutoff radius for inter-monomer short-range edges (Angstrom)
    transform : callable, optional
        Optional transform to apply to data
    pre_transform : callable, optional
        Optional pre-transform to apply during processing
    pre_filter : callable, optional
        Optional filter to apply during processing
    force_reprocess : bool
        Force reprocessing of dataset
    check_monomer_validity : bool
        Check if molecular featurization is valid
    max_size : int, optional
        Maximum number of data points to process
    """
    
    def __init__(
        self,
        root,
        fsapt_dataframe: Optional[pd.DataFrame] = None,
        r_cut=5.0,
        r_cut_im=8.0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reprocess=False,
        check_monomer_validity=True,
        max_size=None,
    ):
        self.fsapt_dataframe = fsapt_dataframe
        self.r_cut = r_cut
        self.r_cut_im = r_cut_im
        self.force_reprocess = force_reprocess
        self.check_monomer_validity = check_monomer_validity
        self.max_size = max_size
        self.data_list = []
        
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            
        super(AP3FusedFSAPTDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        # Load processed data
        self._load_processed_data()
    
    @property
    def raw_file_names(self):
        """Raw file names (not used if dataframe is provided directly)"""
        return ['fsapt_data.pkl']
    
    @property
    def processed_file_names(self):
        """Processed file names"""
        if self.force_reprocess:
            return ['force_reprocess']
        
        processed_path = osp.join(self.processed_dir, 'fsapt_data.pt')
        if osp.exists(processed_path):
            return ['fsapt_data.pt']
        return ['missing']
    
    def download(self):
        """Download or prepare raw data (not needed if dataframe provided)"""
        pass
    
    def process(self):
        """Process FSAPT dataframe into PyTorch Geometric Data objects"""
        if self.fsapt_dataframe is None:
            # Try to load from raw directory
            raw_path = osp.join(self.raw_dir, 'fsapt_data.pkl')
            if osp.exists(raw_path):
                self.fsapt_dataframe = pd.read_pickle(raw_path)
            else:
                raise ValueError(
                    "No FSAPT dataframe provided and no fsapt_data.pkl found in raw directory"
                )
        
        data_list = []
        n_rows = len(self.fsapt_dataframe)
        if self.max_size is not None:
            n_rows = min(n_rows, self.max_size)
        
        print(f"Processing {n_rows} FSAPT data points...")
        
        for i in range(n_rows):
            row = self.fsapt_dataframe.iloc[i]
            
            # Skip rows without valid qcel_molecule
            if not hasattr(row.get('qcel_molecule'), 'get_fragment'):
                print(f"Skipping row {i}: invalid qcel_molecule")
                continue
            
            # Create data object
            data = fsapt_dimer_to_fused_data(
                row=row,
                r_cut=self.r_cut,
                r_cut_im=self.r_cut_im,
                dimer_ind=i,
                check_validity=self.check_monomer_validity,
            )
            
            if data is None:
                print(f"Skipping row {i}: invalid dimer")
                continue
            
            # Apply pre-filter and pre-transform
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        # Save processed data
        processed_path = osp.join(self.processed_dir, 'fsapt_data.pt')
        torch.save(data_list, processed_path)
        print(f"Processed {len(data_list)} data points, saved to {processed_path}")
    
    def _load_processed_data(self):
        """Load processed data from disk"""
        processed_path = osp.join(self.processed_dir, 'fsapt_data.pt')
        if osp.exists(processed_path):
            self.data_list = torch.load(processed_path, weights_only=False)
        else:
            self.data_list = []
    
    def len(self):
        """Return number of data points"""
        return len(self.data_list)
    
    def get(self, idx):
        """Get data point by index"""
        data = self.data_list[idx]
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
