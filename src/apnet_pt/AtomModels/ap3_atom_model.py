import torch
import torch.nn as nn
from ..util import scatter_sum_compile
import numpy as np
import warnings
from .. import multipole
import time
from apnet_pt.atomic_datasets import (
    atomic_module_dataset,
    AtomicDataLoader,
    atomic_collate_update,
    qcel_mon_to_pyg_data,
    atomic_collate_update_no_target,
)
from apnet_pt.multipole import thole_damping_mutual_torch, thole_damping_direct_torch
from apnet_pt import constants

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from importlib import resources
import qcelemental as qcel
from .ap2_atom_model import (
    unsorted_segment_sum_3d,
    get_distances,
    DistanceLayer,
    max_Z,
    isolate_atomic_property_predictions,
    unwrap_model,
)
from .ap3_atomtype_mpnn import AtomTypeParamMPNN

warnings.filterwarnings("ignore")


class AtomInducedDipoleMPNN(torch.nn.Module):
    def __init__(
        self,
        atomtype_hfvr_model=AtomTypeParamMPNN(),
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
    ):
        super().__init__()
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut = r_cut
        self.atomtype_hfvr_model = atomtype_hfvr_model
        self.atomtype_hfvr_model.requires_grad_(False)

        self.polarizability_table = constants.polarizability_table.clone()
        # embed interatomic distances into large orthogonal basis
        self.distance_layer = DistanceLayer(n_rbf, r_cut)

        # embed atom types
        self.embed_layer = nn.Embedding(max_Z + 1, n_embed)

        # zero-th order charge guess, based solely on atom type
        self.guess_layer = nn.Embedding(max_Z + 1, 1)

        # update layers for hidden states
        self.charge_update_layers = nn.ModuleList()
        self.dipole_update_layers = nn.ModuleList()
        self.qpole1_update_layers = nn.ModuleList()
        self.qpole2_update_layers = nn.ModuleList()

        # readout layers for predicting multipoles from hidden states
        self.charge_readout_layers = nn.ModuleList()
        self.dipole_readout_layers = nn.ModuleList()
        self.qpole_readout_layers = nn.ModuleList()

        input_layer_size = n_embed * 4 * n_rbf + n_embed * 4 + n_rbf + 4

        layer_nodes_hidden = [
            input_layer_size,
            n_neuron * 2,
            n_neuron,
            n_neuron // 2,
            n_embed,
        ]
        layer_nodes_readout = [
            n_embed,
            n_neuron * 2,
            n_neuron,
            n_neuron // 2,
            1,
        ]
        layer_activations = [
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            None,
        ]  # None represents a linear activation

        for i in range(n_message):
            self.charge_update_layers.append(
                self._make_layers(layer_nodes_hidden, layer_activations)
            )
            self.dipole_update_layers.append(
                self._make_layers(layer_nodes_hidden, layer_activations)
            )
            self.qpole1_update_layers.append(
                self._make_layers(layer_nodes_hidden, layer_activations)
            )
            self.qpole2_update_layers.append(
                self._make_layers(layer_nodes_hidden, layer_activations)
            )

            self.charge_readout_layers.append(
                self._make_layers(layer_nodes_readout, layer_activations)
            )
            self.dipole_readout_layers.append(nn.Linear(n_embed, 1))
            self.qpole_readout_layers.append(nn.Linear(n_embed, 1))

    def _make_layers(self, layer_nodes, activations):
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            if activations[i] is not None:
                layers.append(activations[i])
        return nn.Sequential(*layers)

    def get_messages(self, h0, h, rbf, hfvr, vw, e_source, e_target):
        nedge = e_source.size(0)

        h0_source = h0.index_select(0, e_source)
        h0_target = h0.index_select(0, e_target)
        h_source = h.index_select(0, e_source)
        h_target = h.index_select(0, e_target)
        # hfvr, vw
        hfvr_source = hfvr.index_select(0, e_source)
        hfvr_target = hfvr.index_select(0, e_target)
        vw_source = vw.index_select(0, e_source)
        vw_target = vw.index_select(0, e_target)
        param_all = torch.cat([hfvr_source, hfvr_target, vw_source, vw_target], dim=-1)
        # [edges x 4 * n_embed]
        h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)

        # [edges, 4 * n_embed, n_rbf]
        h_all_dot = torch.einsum("ez,er->ezr", h_all, rbf)
        # [edges, 4 * n_embed * n_rbf]
        h_all_dot = h_all_dot.view(nedge, -1)
        m_ij = torch.cat([h_all, h_all_dot, rbf, param_all], dim=-1)
        return m_ij

    def monomer_induced_dipole_torch(
        self,
        Z,
        R,
        q,
        mu,
        quad,
        e_source,
        e_target,
        hirshfeld_volume_ratio: torch.Tensor,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-8,
        omega: float = 0.7,
        thole_damping_param_mutual: float = 0.39,
        thole_damping_param_direct: float = 0.34,
        screening: bool = True,
        screening_distance: float = 2.0,
        compute_energies: bool = False,
    ) -> tuple:
        """
        Calculate intramolecular induced dipoles for a single molecule using
        its multipole moments and Hirshfeld volume ratios. This is the PyTorch
        version of the intramolecular_induced_dipole function, following the
        classical induction model from CLIFF paper.

        Reference: https://pubs.aip.org/aip/jcp/article/154/18/184110/200216/CLIFF-A-component-based-machine-learned

        Parameters
        ----------
        Z : torch.Tensor
            Atomic numbers (n_atoms,)
        R : torch.Tensor
            Atomic positions in Bohr (n_atoms, 3)
        q : torch.Tensor
            Atomic charges (n_atoms, 1) or (n_atoms,)
        mu : torch.Tensor
            Atomic dipole moments (n_atoms, 3)
        quad : torch.Tensor
            Atomic quadrupole moments (n_atoms, 3, 3)
        e_source : torch.Tensor
            Source atom indices for intramolecular pairs
        e_target : torch.Tensor
            Target atom indices for intramolecular pairs
        hirshfeld_volume_ratio : torch.Tensor
            Hirshfeld volume ratios for polarizability scaling (n_atoms,)
        max_iterations : int
            Maximum number of SCF iterations (default: 200)
        convergence_threshold : float
            Convergence threshold for induced dipoles (default: 1e-8)
        omega : float
            Damping parameter for SCF convergence (default: 0.7, recommended)
        thole_damping_param_mutual : float
            Thole damping parameter for induced-induced interactions (default: 0.39)
        thole_damping_param_direct : float
            Thole damping parameter for permanent-induced interactions (default: 0.34)
        screening : bool
            Enable distance-based screening for 1-2, 1-3 interactions (default: True)
        screening_distance : float
            Distance threshold in Angstroms for screening (default: 1.8)
        compute_energies : bool
            If True, compute and return intramolecular induction energy (default: False)

        Returns
        -------
        tuple
            (charges, induced_dipoles, quadrupoles) or
            (charges, induced_dipoles, quadrupoles, energy) if compute_energies=True
            - charges: original charges (n_atoms,)
            - induced_dipoles: converged induced dipole moments (n_atoms, 3)
            - quadrupoles: original quadrupoles (n_atoms, 3, 3)
            - energy (optional): intramolecular induction energy in kcal/mol
        """
        from apnet_pt.AtomPairwiseModels.mtp_mtp import get_distances

        h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

        # Calculate atomic polarizabilities
        alpha_0 = torch.index_select(self.polarizability_table, 0, Z.long())
        alpha = alpha_0 * hirshfeld_volume_ratio ** (4 / 3.0)

        # Define helper function to calculate distance tensors with Thole damping
        def distance_tensors(
            Ri,
            Rj,
            e_source,
            e_target,
            alpha_i,
            alpha_j,
            thole_param,
            apply_screening=False,
        ):
            """Calculate interaction tensors between atoms with optional screening"""
            dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
            dR_xyz = dR_xyz_ang / constants.au2ang
            dR = dR_ang / constants.au2ang

            alpha_source = alpha_i.index_select(0, e_source)
            alpha_target = alpha_j.index_select(0, e_target)

            # Apply Thole damping
            if apply_screening:
                au3, lam_3, lam_5 = thole_damping_direct_torch(
                    dR, alpha_source, alpha_target, thole_param
                )
            else:
                au3, lam_3, lam_5 = thole_damping_mutual_torch(
                    dR, alpha_source, alpha_target, thole_param
                )

            # Apply distance-based screening for direct interactions (exclude 1-2, 1-3 bonds)
            if apply_screening and screening:
                screening_mask = dR_ang < screening_distance
                lam_3 = torch.where(screening_mask, torch.zeros_like(lam_3), lam_3)
                lam_5 = torch.where(screening_mask, torch.zeros_like(lam_5), lam_5)
                dR = torch.where(screening_mask, torch.ones_like(dR), dR)

            delta = torch.eye(3, device=dR.device)
            oodR = 1.0 / dR

            # T1: field tensor (rank 1)
            # Note: dR_xyz points FROM source TO target, which is the correct direction
            # for the field at target due to source. No negation needed.
            T1 = torch.einsum("x,xy,x->xy", oodR**3, dR_xyz, lam_3)

            # T2: field gradient tensor (rank 2)
            T2 = 3 * torch.einsum("xy,xz,x->xyz", dR_xyz, dR_xyz, lam_5) - torch.einsum(
                "x,x,yz,x->xyz", dR, dR, delta, lam_3
            )
            T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)

            return dR, dR_xyz, oodR, T1, T2

        # Calculate direct tensors (permanent → induced) with screening
        dR_direct, dR_xyz_direct, T0_direct, T1_direct, T2_direct = distance_tensors(
            R,
            R,
            e_source,
            e_target,
            alpha,
            alpha,
            thole_damping_param_direct,
            apply_screening=True,
        )

        # Calculate mutual tensors (induced ↔ induced) without screening
        dR_mutual, dR_xyz_mutual, T0_mutual, T1_mutual, T2_mutual = distance_tensors(
            R,
            R,
            e_source,
            e_target,
            alpha,
            alpha,
            thole_damping_param_mutual,
            apply_screening=False,
        )

        # Initialize induced dipoles
        # print(f"{R.size() = }")
        n_atoms = R.shape[0]
        mu_induced_0 = torch.zeros((n_atoms, 3), device=q.device)

        # Select relevant tensors for atom pairs
        # alpha_source = alpha.index_select(0, e_source)
        alpha_target = alpha.index_select(0, e_target)
        q_source = q.squeeze(-1).index_select(0, e_source)
        mu_source = mu.index_select(0, e_source)

        # Calculate initial induced dipoles from permanent multipoles (using direct tensors)
        # Contribution from charges: mu_ind = alpha * T1 * q
        mu_charge = torch.einsum("a,ai,a->ai", alpha_target, T1_direct, q_source)
        mu_induced_0 = scatter_sum_compile(mu_charge, e_target, n_atoms)

        # Contribution from dipoles: mu_ind += alpha * T2 * mu
        mu_dipole = torch.einsum("a,aij,aj->ai", alpha_target, T2_direct, mu_source)
        mu_dipole_summed = scatter_sum_compile(mu_dipole, e_target, n_atoms)
        mu_induced_0 += mu_dipole_summed

        # print(f"{mu_induced_0.size()}")
        # print(f"{Z.size()}")
        # print(f"{Z = }")
        # print(f"{e_source = }")
        # print(f"{e_target = }")
        # apply heavy_atoms_only logic in torch
        mu_induced_0 = torch.where(
            Z.unsqueeze(-1) == 1, torch.zeros_like(mu_induced_0), mu_induced_0
        )
        Z_source = Z.index_select(0, e_source)
        Z_target = Z.index_select(0, e_target)
        # set all T tenors to zero where either source or target is hydrogen
        hydrogen_mask = (Z_source == 1) | (Z_target == 1)
        T2_mutual = torch.where(
            hydrogen_mask.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(T2_mutual),
            T2_mutual,
        )
        T1_direct = torch.where(
            hydrogen_mask.unsqueeze(-1),
            torch.zeros_like(T1_direct),
            T1_direct,
        )

        # Self-consistent field (SCF) iteration to converge induced dipoles
        mu_induced = mu_induced_0.clone()

        for iteration in range(max_iterations):
            mu_induced_old = mu_induced.clone()

            # Induced dipoles due to other induced dipoles (using mutual tensors)
            mu_induced_contrib = torch.einsum(
                "a,aij,aj->ai",
                alpha_target,
                T2_mutual,
                mu_induced.index_select(0, e_source),
            )
            mu_induced_new = scatter_sum_compile(mu_induced_contrib, e_target, n_atoms)
            # Add initial induced dipoles from permanent multipoles
            mu_induced_new += mu_induced_0

            # Apply mixing for numerical stability
            mu_induced = (1 - omega) * mu_induced_old + omega * mu_induced_new

            # Check convergence
            delta = torch.norm(mu_induced - mu_induced_old)
            if delta < convergence_threshold:
                break

        return mu_induced

    # @torch.jit.trace
    def forward(
        self,
        batch,
    ):
        # Extract variables from batch
        x = batch.x
        edge_index = batch.edge_index
        R = batch.R
        molecule_ind = batch.molecule_ind
        total_charge = batch.total_charge
        natom_per_mol = batch.natom_per_mol

        # edge_index has shape [(e_source, e_target), n_edges]
        Z = x
        natom = Z.size(0)

        h_list_0 = [self.embed_layer(Z)]

        # Initial guesses
        charge = self.guess_layer(Z)

        dipole = torch.zeros(natom, 3, dtype=torch.float32, device=Z.device)
        qpole = torch.zeros(natom, 3, 3, dtype=torch.float32, device=Z.device)

        if edge_index.size(1) == 0:
            # need h_list to have the same number of dimensions as the number of message passing layers
            h_list = [h_list_0[0] for i in range(self.n_message + 1)]
            h_list = torch.stack(h_list, dim=1)
            molecule_ind.requires_grad_(False)
            molecule_ind = molecule_ind.long()
            num_mols = (
                int(molecule_ind.max().item()) + 1 if molecule_ind.numel() > 0 else 1
            )
            total_charge_pred = scatter_sum_compile(
                charge, molecule_ind, num_mols, reduce="sum"
            )
            total_charge_pred = total_charge_pred.squeeze()
            total_charge_err = total_charge_pred - total_charge
            charge_err = torch.repeat_interleave(
                total_charge_err / natom_per_mol.float(), natom_per_mol
            ).unsqueeze(1)
            charge = charge - charge_err
            return charge, dipole, qpole, h_list

        Ks = self.atomtype_hfvr_model(batch)
        hfvr = Ks[:, 0].unsqueeze(1)
        vw = Ks[:, 1].unsqueeze(1)
        # 1) Filter out atoms that don't have edges
        # Create keep_mask directly from edge_index without using torch.isin
        # This is more compile-friendly than torch.isin with unbacked symbolic shapes
        natom = len(molecule_ind)
        keep_mask = torch.zeros(natom, dtype=torch.bool, device=molecule_ind.device)
        if edge_index.size(1) > 0:
            # Mark all atoms that appear in edge_index as True
            keep_mask.scatter_(0, edge_index[0], True)
            keep_mask.scatter_(0, edge_index[1], True)
        filtered_charge = charge[keep_mask]

        # Now `filtered_charge` contains only atoms from molecules that have >= 2 atoms and edges
        h_list = [h_list_0[0][keep_mask]]

        # Now we need to filter the edge_index to only include edges between
        # atoms in molecules with >= 2 atoms.
        e_source = edge_index[0]
        e_target = edge_index[1]
        edge_keep = keep_mask[e_source] & keep_mask[e_target]
        e_source = e_source[edge_keep]
        e_target = e_target[edge_keep]
        # shape [N], each kept atom -> new index
        idx_map = torch.cumsum(keep_mask, dim=0) - 1
        idx_map = idx_map.long()  # ensure integer
        e_source = idx_map[e_source]
        e_target = idx_map[e_target]

        R_mask = R[keep_mask, :]
        natom_filtered = keep_mask.sum()

        #  [edges]
        dR, dR_xyz = get_distances(R_mask, R_mask, e_source, e_target)

        # [edges x 3]
        dr_unit = dR_xyz / dR.unsqueeze(1)
        rbf = self.distance_layer(dR)

        for i in range(self.n_message):
            #####################
            ### charge update ###
            #####################

            # [edges x message_embedding_dim]
            m_ij = self.get_messages(
                h_list[0], h_list[-1], rbf, hfvr, vw, e_source, e_target
            )

            # [atoms x message_embedding_dim]
            m_i = scatter_sum_compile(m_ij, e_source, int(natom_filtered), reduce="sum")  # type: ignore

            # [atomx x hidden_dim]
            h_next = self.charge_update_layers[i](m_i)
            h_list.append(h_next)
            charge_update = self.charge_readout_layers[i](h_list[i + 1])
            filtered_charge += charge_update

            #####################
            ### dipole update ###
            #####################

            # [edges x n_embed]
            m_ij_dipole = self.dipole_update_layers[i](m_ij)
            # [edges x 3 x n_embed]
            m_ij_dipole = torch.einsum("ex,em->exm", dr_unit, m_ij_dipole)
            # [atoms x 3 x n_embed]
            m_i_dipole = unsorted_segment_sum_3d(m_ij_dipole, e_source, natom)
            # [atoms x 3 x 1]
            d_dipole = self.dipole_readout_layers[i](m_i_dipole)
            # [atoms x 3]
            d_dipole = d_dipole.view(natom, 3)
            dipole += d_dipole

            #########################
            ### quadrupole update ###
            #########################

            # [edges x n_embed]
            m_ij_qpole1 = self.qpole1_update_layers[i](m_ij)
            # [edges x 3 x n_embed]
            m_ij_qpole1 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole1)
            # [atoms x 3 x n_embed]
            m_i_qpole1 = unsorted_segment_sum_3d(m_ij_qpole1, e_source, natom)

            # [edges x n_embed]
            m_ij_qpole2 = self.qpole2_update_layers[i](m_ij)
            # [edges x 3 x n_embed]
            m_ij_qpole2 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole2)
            # [atoms x 3 x n_embed]
            m_i_qpole2 = unsorted_segment_sum_3d(m_ij_qpole2, e_source, natom)
            d_qpole = torch.einsum("axf,ayf->axyf", m_i_qpole1, m_i_qpole2)
            d_qpole = d_qpole + d_qpole.permute(0, 2, 1, 3)
            # Paper states 0.5 factor is applied to the sum
            # d_qpole = 0.5 * (d_qpole + d_qpole.permute(0, 2, 1, 3))
            d_qpole = self.qpole_readout_layers[i](d_qpole)
            d_qpole = d_qpole.view(natom, 3, 3)
            qpole += d_qpole

        ####################################
        ### enforce traceless quadrupole ###
        ####################################

        qpole = multipole.ensure_traceless_qpole(qpole)

        ###################################
        ### enforce charge conservation ###
        ###################################

        charge[keep_mask] = filtered_charge
        molecule_ind.requires_grad_(False)
        molecule_ind = molecule_ind.long()
        num_mols = int(molecule_ind.max().item()) + 1 if molecule_ind.numel() > 0 else 1
        total_charge_pred = scatter_sum_compile(
            charge, molecule_ind, num_mols, reduce="sum"
        )
        # return charge, dipole, qpole, h_list

        total_charge_pred = total_charge_pred.squeeze()
        total_charge_err = total_charge_pred - total_charge
        charge_err = torch.repeat_interleave(
            total_charge_err / natom_per_mol.float(), natom_per_mol
        ).unsqueeze(1)
        charge = charge - charge_err
        charge = charge.squeeze()
        # changed to dim=0 from dim=1 for usage in Param fitting # AMW 8/20/25
        # Breaks test_apnet2_train_qcel_molecules_in_memory_transfer test,
        # dimensions no longer correct... figure out another way to fix this. reverting back to dim=1 # AMW 9/17/25
        h_list = torch.stack(h_list, dim=1)

        induced_dipoles = self.monomer_induced_dipole_torch(
            Z,
            R,
            charge.unsqueeze(1),
            dipole,
            qpole,
            e_source,
            e_target,
            hirshfeld_volume_ratio=Ks[:, 0],
        )
        dipole += induced_dipoles
        return charge, dipole, qpole, h_list


class AtomInducedDipoleModel:
    def __init__(
        self,
        dataset=None,
        pre_trained_model_path=None,
        atomtype_hfvr_model=None,
        atomtype_hfvr_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data",
        ds_max_size=None,
        ds_testing=False,
        ds_force_reprocess=False,
        ds_in_memory=True,
        model_save_path=None,
    ):
        """
        If pre_trained_model_path is provided, the model will be loaded from
        the path and all other parameters will be ignored except for dataset.

        use_GPU will check for a GPU and use it if available unless set to false.
        """
        if atomtype_hfvr_model is None and atomtype_hfvr_pre_trained_path is None:
            raise ValueError(
                "Either atomtypeparam_hfvr_model or atomtypeparam_hfvr_pre_trained_path must be provided.\n"
                "Without a model predicting hirshfeld volumes, induced dipoles cannot be computed correctly"
            )
        if torch.cuda.is_available() and use_GPU is not False:
            device = torch.device("cuda:0")
            print("running on the GPU")
        else:
            device = torch.device("cpu")
            print("running on the CPU")

        if atomtype_hfvr_model is not None:
            self.atomtype_hfvr_model = atomtype_hfvr_model

        if atomtype_hfvr_pre_trained_path is not None:
            print(
                f"Loading pre-trained AtomTypeParamMPNN hfvr model from {atomtype_hfvr_pre_trained_path}"
            )
            checkpoint = torch.load(atomtype_hfvr_pre_trained_path, weights_only=False)
            self.atomtype_hfvr_model = AtomTypeParamMPNN(
                n_message=checkpoint["config"]["n_message"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                param_start_mean=checkpoint["config"]["param_start_mean"],
                param_start_std=checkpoint["config"]["param_start_std"],
                n_params=checkpoint["config"].get("n_params", 1),
                r_cut=checkpoint["config"]["r_cut"],
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.atomtype_hfvr_model.load_state_dict(model_state_dict)

        if pre_trained_model_path:
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = AtomInducedDipoleMPNN(
                atomtype_hfvr_model=self.atomtype_hfvr_model,
                n_message=checkpoint["config"]["n_message"],
                n_rbf=checkpoint["config"]["n_rbf"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                r_cut=checkpoint["config"]["r_cut"],
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = AtomInducedDipoleMPNN(
                atomtype_hfvr_model=self.atomtype_hfvr_model,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut=r_cut,
            )
        self.device = device
        self.dataset = dataset
        self.ds_spec_type = ds_spec_type
        mp.set_sharing_strategy("file_system")
        split_dbs = [7]
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
        ):
            print("Setting up dataset...")

            def setup_ds(fp=ds_force_reprocess):
                return atomic_module_dataset(
                    root=ds_root,
                    testing=ds_testing,
                    spec_type=ds_spec_type,
                    max_size=ds_max_size,
                    force_reprocess=fp,
                    in_memory=ds_in_memory,
                )

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
        elif (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type in split_dbs
        ):
            print("Processing Split dataset...")

            def setup_ds(fp=ds_force_reprocess):
                return [
                    atomic_module_dataset(
                        root=ds_root,
                        testing=ds_testing,
                        spec_type=ds_spec_type,
                        split="train",
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        in_memory=ds_in_memory,
                    ),
                    atomic_module_dataset(
                        root=ds_root,
                        testing=ds_testing,
                        spec_type=ds_spec_type,
                        split="test",
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        in_memory=ds_in_memory,
                    ),
                ]

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
        print(f"{self.dataset = }")
        self.rank = None
        self.world_size = None
        self.model_save_path = model_save_path
        self.train_shuffle = None
        # torch.jit.enable_onednn_fusion(True)
        return

    def set_pretrained_model(self, model_path=None, model_id=None):
        if model_id is not None:
            # model_path = f"{file_dir}/../models/am_ensemble/am_{model_id}.pt"
            model_path = resources.files("apnet_pt").joinpath(
                "models", "am_ensemble", f"am_{model_id}.pt"
            )
        elif model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided.")

        checkpoint = torch.load(model_path, weights_only=False)
        # pp(checkpoint)
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return self

    def compile_model(self):
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        self.model = torch.compile(self.model, dynamic=True)
        return

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        # torch.manual_seed(42)

    def cleanup(self):
        dist.destroy_process_group()

    def _qcel_example_input(self, mols, batch_size=1):
        mol_data = [qcel_mon_to_pyg_data(mol) for mol in mols]
        batches = []
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            batch_A = atomic_collate_update_no_target(batch_mol_data)
            batches.append(batch_A)
        return batches

    def example_input(self):
        mol = qcel.models.Molecule.from_data("""
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
units angstrom
        """)
        return self._qcel_example_input([mol], batch_size=1)

    def evaluate_model_collate_train(self, data_loader, optimizer=None, loss_fn=None):
        charge_errors_t, dipole_errors_t, qpole_errors_t = [], [], []
        total_loss = 0.0
        self.model.train()
        for batch in data_loader:
            batch_loss = 0.0
            batch = batch.to(self.device)
            optimizer.zero_grad()
            charge, dipole, qpole, _ = self.model(batch)

            # Errors
            q_error = charge - batch.charges
            d_error = dipole - batch.dipoles
            qp_error = qpole - batch.quadrupoles
            if loss_fn is None:
                # perform mean squared error
                charge_loss = torch.mean(torch.square(q_error))
                dipole_loss = torch.mean(torch.square(d_error))
                qpole_loss = torch.mean(torch.square(qp_error))
            else:
                # perform custom loss function, or pytorch criterion loss_fn
                charge_loss = loss_fn(charge, batch.charges)
                dipole_loss = torch.mean(loss_fn(dipole, batch.dipoles))
                qpole_loss = torch.mean(loss_fn(qpole, batch.quadrupoles))

            batch_loss = charge_loss + dipole_loss + qpole_loss
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.detach().item()

            charge_errors_t.append(q_error.detach())
            dipole_errors_t.extend(d_error.detach())
            qpole_errors_t.extend(qp_error.detach())
        charge_errors_t = torch.cat(charge_errors_t)
        dipole_errors_t = torch.cat(dipole_errors_t)
        qpole_errors_t = torch.cat(qpole_errors_t)
        return total_loss, charge_errors_t, dipole_errors_t, qpole_errors_t

    def evaluate_model_collate_eval(self, data_loader, loss_fn=None):
        charge_errors_t, dipole_errors_t, qpole_errors_t = [], [], []
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                # print('saving batch for debug')
                # torch.save(batch, "debug_batch.pt")
                batch_loss = 0.0
                batch = batch.to(self.device)
                charge, dipole, qpole, hlist = self.model(batch)

                # Errors
                q_error = charge - batch.charges
                d_error = dipole - batch.dipoles
                qp_error = qpole - batch.quadrupoles
                if loss_fn is None:
                    # perform mean squared error
                    charge_loss = torch.mean(torch.square(q_error))
                    dipole_loss = torch.mean(torch.square(d_error))
                    qpole_loss = torch.mean(torch.square(qp_error))
                else:
                    # perform custom loss function, or pytorch criterion loss_fn
                    charge_loss = loss_fn(charge, batch.charges)
                    dipole_loss = torch.mean(loss_fn(dipole, batch.dipoles))
                    qpole_loss = torch.mean(loss_fn(qpole, batch.quadrupoles))

                batch_loss = charge_loss + dipole_loss + qpole_loss
                total_loss += batch_loss.detach()

            charge_errors_t.append(q_error.detach().cpu())
            dipole_errors_t.extend(d_error.detach().cpu())
            qpole_errors_t.extend(qp_error.detach().cpu())
        charge_errors_t = torch.cat(charge_errors_t)
        dipole_errors_t = torch.cat(dipole_errors_t)
        qpole_errors_t = torch.cat(qpole_errors_t)
        return total_loss, charge_errors_t, dipole_errors_t, qpole_errors_t

    def pretrain_statistics(self, train_loader, test_loader, criterion):
        t1 = time.time()
        with torch.no_grad():
            _, charge_errors_t, dipole_errors_t, qpole_errors_t = (
                self.evaluate_model_collate_eval(
                    train_loader,  # loss_fn=criterion
                )
            )
            charge_MAE_t = np.mean(np.abs(charge_errors_t.numpy()))
            dipole_MAE_t = np.mean(np.abs(dipole_errors_t.numpy()))
            qpole_MAE_t = np.mean(np.abs(qpole_errors_t.numpy()))

            charge_errors_t, dipole_errors_t, qpole_errors_t = [], [], []
            test_loss, charge_errors_v, dipole_errors_v, qpole_errors_v = (
                self.evaluate_model_collate_eval(
                    test_loader,  # loss_fn=criterion
                )
            )
            charge_MAE_v = np.mean(np.abs(charge_errors_v.numpy()))
            dipole_MAE_v = np.mean(np.abs(dipole_errors_v.numpy()))
            qpole_MAE_v = np.mean(np.abs(qpole_errors_v.numpy()))
            charge_errors_v, dipole_errors_v, qpole_errors_v = [], [], []
            dt = time.time() - t1
            print(
                f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} {dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f}",
                flush=True,
            )
        return test_loss

    def train_batches_single_proc(
        self, rank, dataloader, criterion, optimizer, rank_device
    ):
        self.model.train()
        total_charge_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_dipole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_qpole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        for batch in dataloader:
            batch = batch.to(rank_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # print('saving batch for debug')
            # torch.save(batch, "debug_batch.pt")
            charge, dipole, qpole, _ = self.model(batch)

            q_error = charge - batch.charges
            d_error = dipole - batch.dipoles
            qp_error = qpole - batch.quadrupoles

            charge_loss = (q_error**2).mean()
            dipole_loss = (d_error**2).mean()
            qpole_loss = (qp_error**2).mean()

            loss = charge_loss + dipole_loss + qpole_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += q_error.numel()

            total_charge_error += q_error.abs().sum()
            total_dipole_error += d_error.abs().sum()
            total_qpole_error += qp_error.abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        charge_mae = total_charge_error.item() / final_count
        dipole_mae = total_dipole_error.item() / (final_count * 3)
        qpole_mae = total_qpole_error.item() / (final_count * 9)
        return total_loss, charge_mae, dipole_mae, qpole_mae

    def train_batches(self, rank, dataloader, criterion, optimizer, rank_device):
        self.model.train()
        total_charge_error = 0
        total_dipole_error = 0
        total_qpole_error = 0
        total_loss = 0
        count = 0

        for batch in dataloader:
            batch = batch.to(rank_device)
            optimizer.zero_grad()
            charge, dipole, qpole, _ = self.model(batch)

            q_error = charge - batch.charges
            d_error = dipole - batch.dipoles
            qp_error = qpole - batch.quadrupoles

            charge_loss = torch.mean(torch.square(q_error))
            dipole_loss = torch.mean(torch.square(d_error))
            qpole_loss = torch.mean(torch.square(qp_error))

            loss = charge_loss + dipole_loss + qpole_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += q_error.numel()

            total_charge_error += torch.sum(torch.abs(q_error)).item()
            total_dipole_error += torch.sum(torch.abs(d_error)).item()
            total_qpole_error += torch.sum(torch.abs(qp_error)).item()

        # Converting to tensors for all-reduce
        total_charge_error = torch.tensor(
            total_charge_error, dtype=torch.float32, device=rank_device
        )
        total_dipole_error = torch.tensor(
            total_dipole_error, dtype=torch.float32, device=rank_device
        )
        total_qpole_error = torch.tensor(
            total_qpole_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_charge_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_dipole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_qpole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        charge_mae = total_charge_error.item() / count.item()
        dipole_mae = total_dipole_error.item() / (count.item() * 3)
        qpole_mae = total_qpole_error.item() / (count.item() * 9)

        return total_loss, charge_mae, dipole_mae, qpole_mae

    def evaluate_batches_single_proc(self, rank, dataloader, criterion, rank_device):
        self.model.eval()
        total_charge_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_dipole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_qpole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        with torch.no_grad():
            for batch in dataloader:
                # print('saving batch for debug')
                # torch.save(batch, "debug_batch.pt")
                batch = batch.to(rank_device, non_blocking=True)
                charge, dipole, qpole, _ = self.model(batch)

                q_error = charge - batch.charges
                d_error = dipole - batch.dipoles
                qp_error = qpole - batch.quadrupoles

                charge_loss = (q_error**2).mean()
                dipole_loss = (d_error**2).mean()
                qpole_loss = (qp_error**2).mean()

                loss = charge_loss + dipole_loss + qpole_loss
                total_loss += loss.item()
                total_count += q_error.numel()

                total_charge_error += q_error.abs().sum()
                total_dipole_error += d_error.abs().sum()
                total_qpole_error += qp_error.abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        charge_mae = total_charge_error.item() / final_count
        dipole_mae = total_dipole_error.item() / (final_count * 3)
        qpole_mae = total_qpole_error.item() / (final_count * 9)
        return total_loss, charge_mae, dipole_mae, qpole_mae

    def evaluate_batches(self, rank, dataloader, criterion, rank_device):
        self.model.eval()
        total_charge_error = 0
        total_dipole_error = 0
        total_qpole_error = 0
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                # print('saving batch for debug')
                # torch.save(batch, "debug_batch.pt")
                batch = batch.to(rank_device)
                charge, dipole, qpole, _ = self.model(batch)

                q_error = charge - batch.charges
                d_error = dipole - batch.dipoles
                qp_error = qpole - batch.quadrupoles

                total_charge_error += torch.sum(torch.abs(q_error)).item()
                total_dipole_error += torch.sum(torch.abs(d_error)).item()
                total_qpole_error += torch.sum(torch.abs(qp_error)).item()

                charge_loss = torch.mean(torch.square(q_error))
                dipole_loss = torch.mean(torch.square(d_error))
                qpole_loss = torch.mean(torch.square(qp_error))

                total_loss += charge_loss + dipole_loss + qpole_loss
                count += q_error.numel()

        # Converting to tensors for all-reduce
        total_charge_error = torch.tensor(
            total_charge_error, dtype=torch.float32, device=rank_device
        )
        total_dipole_error = torch.tensor(
            total_dipole_error, dtype=torch.float32, device=rank_device
        )
        total_qpole_error = torch.tensor(
            total_qpole_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_charge_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_dipole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_qpole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_loss = torch.tensor(total_loss.item(), device=rank_device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        charge_mae = total_charge_error.item() / count.item()
        dipole_mae = total_dipole_error.item() / (count.item() * 3)
        qpole_mae = total_qpole_error.item() / (count.item() * 9)

        return total_loss, charge_mae, dipole_mae, qpole_mae

    def ddp_train(
        self,
        rank,
        world_size,
        train_dataset,
        test_dataset,
        n_epochs,
        batch_size,
        lr,
        pin_memory,
        num_workers,
    ):
        if self.device.type == "cpu":
            # rank = "cpu"
            rank_device = "cpu"
        else:
            rank_device = rank
        if world_size > 1:
            self.setup(rank, world_size)

        self.model.to(rank_device)
        if world_size > 1 and rank_device == "cpu":
            torch._dynamo.config.dynamic_shapes = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, dynamic=True)
            self.model = DDP(
                self.model,
            )
        # elif rank_device != "cpu":
        #     self.model = DDP(
        #         self.model,
        #         device_ids=[rank],
        #         output_device=rank_device,
        #     )

        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank
            )
            if world_size > 1
            else None
        )
        test_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if world_size > 1
            else None
        )

        train_loader = AtomicDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=atomic_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=atomic_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        test_loss = self.pretrain_statistics(train_loader, test_loader, criterion)

        lowest_test_loss = test_loss

        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, charge_MAE_t, dipole_MAE_t, qpole_MAE_t = self.train_batches(
                rank, train_loader, criterion, optimizer, rank_device
            )
            test_loss, charge_MAE_v, dipole_MAE_v, qpole_MAE_v = self.evaluate_batches(
                rank, test_loader, criterion, rank_device
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        # cpu_model = self.model.to("cpu")
                        cpu_model = unwrap_model(self.model).to("cpu")
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_rbf": cpu_model.n_rbf,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "r_cut": cpu_model.r_cut,
                                },
                            },
                            self.model_save_path,
                        )
                        self.model.to(self.device)
                else:
                    test_lowered = " "
                dt = time.time() - t1
                test_loss = 0.0
                # if (world_size==1 or rank == 0):
                print(
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} {dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f} {test_lowered}",
                    flush=True,
                )
        if world_size > 1:
            self.cleanup()
        return

    def single_proc_train(
        self,
        rank,
        world_size,
        train_dataset,
        test_dataset,
        n_epochs,
        batch_size,
        lr,
        pin_memory,
        num_workers,
        skip_compile=True,
    ):
        if self.device.type == "cpu":
            rank_device = "cpu"
        else:
            rank_device = rank

        self.model.to(rank_device)
        if not skip_compile:
            self.compile_model()

        train_loader = AtomicDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=self.train_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=atomic_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=atomic_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        lowest_test_loss = torch.tensor(float("inf"))
        test_loss = self.pretrain_statistics(train_loader, test_loader, criterion)

        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, charge_MAE_t, dipole_MAE_t, qpole_MAE_t = (
                self.train_batches_single_proc(
                    rank, train_loader, criterion, optimizer, rank_device
                )
            )
            test_loss, charge_MAE_v, dipole_MAE_v, qpole_MAE_v = (
                self.evaluate_batches_single_proc(
                    rank, test_loader, criterion, rank_device
                )
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        # cpu_model = self.model.to("cpu")
                        cpu_model = unwrap_model(self.model).to("cpu")
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_rbf": cpu_model.n_rbf,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "r_cut": cpu_model.r_cut,
                                },
                            },
                            self.model_save_path,
                        )
                        self.model.to(self.device)
                else:
                    test_lowered = " "
                dt = time.time() - t1
                test_loss = 0.0
                print(
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} {dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f} {test_lowered}",
                    flush=True,
                )
        if world_size > 1:
            self.cleanup()
        return

    def train(
        self,
        dataset=None,
        n_epochs=500,
        batch_size=16,
        lr=5e-4,
        split_percent=0.9,
        model_path=None,
        skip_compile=False,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,  # Default to 1 for single-core operation
        omp_num_threads_per_process=None,
        random_seed=42,
    ):
        self.model_save_path = model_path
        if self.model_save_path is not None:
            print(f"Saving model to {self.model_save_path}")
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        elif dataset is not None:
            print("Overriding self.dataset with passed dataset!")
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided")
        self.train_shuffle = shuffle

        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if isinstance(self.dataset, list):
            train_dataset = self.dataset[0]
            if shuffle:
                order_indices = np.random.permutation(len(train_dataset))
            else:
                order_indices = [i for i in range(len(train_dataset))]
            train_dataset = train_dataset[order_indices]

            test_dataset = self.dataset[1]
            if shuffle:
                order_indices = np.random.permutation(len(test_dataset))
            else:
                order_indices = [i for i in range(len(test_dataset))]
            test_dataset = test_dataset[order_indices]
        else:
            if shuffle:
                order_indices = np.random.permutation(len(self.dataset))
            else:
                order_indices = np.arange(len(self.dataset))
            train_indices = order_indices[: int(len(self.dataset) * split_percent)]
            test_indices = order_indices[int(len(self.dataset) * split_percent) :]
            train_dataset = self.dataset[train_indices]
            test_dataset = self.dataset[test_indices]

        print("~~ Training Atom Model ~~", flush=True)
        print(
            f"    Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples",
            flush=True,
        )
        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {self.model.n_message=}", flush=True)
        print(f"  {self.model.n_neuron=}", flush=True)
        print(f"  {self.model.n_embed=}", flush=True)
        print(f"  {self.model.n_rbf=}", flush=True)
        print(f"  {self.model.r_cut=}", flush=True)
        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {batch_size=}", flush=True)
        print(f"  {lr=}\n", flush=True)

        # pin_memory = torch.cuda.is_available()
        pin_memory = True

        if skip_compile:
            torch.jit.enable_onednn_fusion(True)
            torch.autograd.set_detect_anomaly(False)

        if world_size > 1:
            # os.environ["OMP_NUM_THREADS"] = str(dataloader_num_workers + 1)
            print("Running multi-process training", flush=True)
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads_per_process)
            mp.spawn(
                self.ddp_train,
                args=(
                    world_size,
                    train_dataset,
                    test_dataset,
                    n_epochs,
                    batch_size,
                    lr,
                    pin_memory,
                    dataloader_num_workers,
                ),
                nprocs=world_size,
                join=True,
            )
        else:
            # Run single-process training directly
            print("Running single-process training", flush=True)
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads_per_process)
            self.single_proc_train(
                rank=0,
                world_size=world_size,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                pin_memory=pin_memory,
                num_workers=dataloader_num_workers,
                skip_compile=skip_compile,
            )

        return

    @torch.inference_mode()
    def predict_multipoles_batch(self, batch, isolate_predictions=True):
        batch.to(self.device)
        self.model.to(self.device)
        qA, muA, thA, hlistA = self.model(batch)
        batch = batch.cpu()
        qA = qA.detach().detach().cpu()
        muA = muA.detach().detach().cpu()
        thA = thA.detach().detach().cpu()
        hlistA = hlistA.detach().cpu()
        if isolate_predictions:
            return isolate_atomic_property_predictions(batch, (qA, muA, thA, hlistA))
        else:
            return qA, muA, thA, hlistA

    @torch.inference_mode()
    def predict_multipoles_dataset(
        self,
        batch_size=16,
        dataloader_num_workers=0,
        world_size=1,  # Default to 1 for single-process operation
        # omp_num_threads_per_process=None,
    ):
        output = []
        data = AtomicDataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        if world_size > 1:
            raise NotImplementedError(
                "Multi-process prediction not implemented yet due to needing to determine how to handle the output data merging."
            )
            # output = mp.spawn(
            #     self.predict_multipoles_dataset_process,
            #     args=(data, batch_size, dataloader_num_workers),
            #     nprocs=world_size,
            #     join=True,
            # )
        else:
            for batch in data:
                charges, dipoles, qpoles, hlists = self.model_predict(batch)
                # need to use batch.molecule_ind to reassemble the output
                mol_charges = [[] for i in range(batch_size)]
                mol_dipoles = [[] for i in range(batch_size)]
                mol_qpoles = [[] for i in range(batch_size)]
                for n, i in enumerate(batch.molecule_ind):
                    mol_charges[i].append(charges[n])
                    mol_dipoles[i].append(dipoles[n])
                    mol_qpoles[i].append(qpoles[n])
                output.append((mol_charges, mol_dipoles, mol_qpoles, hlists))
        return output

    @torch.inference_mode()
    def predict_qcel_mols(self, mols, batch_size=2):
        output = []
        mol_data = []
        cnt = 0
        for mol in mols:
            data = qcel_mon_to_pyg_data(mol)
            mol_data.append(data)
            cnt += 1
            if len(mol_data) == batch_size or cnt == len(mols):
                batch = atomic_collate_update_no_target(mol_data)
                with torch.no_grad():
                    charge, dipole, qpole, hlist = self.model(batch)
                    # Isolate atomic properties by molecule
                    mol_charges, mol_dipoles, mol_qpoles, mol_hlists = (
                        isolate_atomic_property_predictions(
                            batch, (charge, dipole, qpole, hlist)
                        )
                    )
                    output.extend(
                        list(zip(mol_charges, mol_dipoles, mol_qpoles, mol_hlists))
                    )
                mol_data = []
        return output

    @torch.inference_mode()
    def model_predict(self, data):
        charge, dipole, qpole, hlist = self.model(
            data.x,
            data.edge_index,
            # data.edge_attr,
            R=data.R,
            molecule_ind=data.molecule_ind,
            total_charge=data.total_charge,
            natom_per_mol=data.natom_per_mol,
        )
        return charge, dipole, qpole, hlist
