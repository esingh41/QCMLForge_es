import torch
import numpy as np
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_collate_update,
    ap3_fused_collate_update_no_target,
    ap3_fused_collate_update_no_target_monomer_indices,
)
from torch_geometric.data import Data


def create_simple_test_data(natoms_A=3, natoms_B=3, n_sr_edges=5, n_lr_edges=3, dimer_id=0):
    RA = torch.randn(natoms_A, 3)
    ZA = torch.randint(1, 10, (natoms_A,))
    RB = torch.randn(natoms_B, 3)
    ZB = torch.randint(1, 10, (natoms_B,))
    
    e_ABsr_source = torch.randint(0, natoms_A, (n_sr_edges,))
    e_ABsr_target = torch.randint(0, natoms_B, (n_sr_edges,))
    e_ABlr_source = torch.randint(0, natoms_A, (n_lr_edges,))
    e_ABlr_target = torch.randint(0, natoms_B, (n_lr_edges,))
    
    e_AA_source = torch.randint(0, natoms_A, (natoms_A,))
    e_AA_target = torch.randint(0, natoms_A, (natoms_A,))
    e_BB_source = torch.randint(0, natoms_B, (natoms_B,))
    e_BB_target = torch.randint(0, natoms_B, (natoms_B,))
    
    dimer_ind = torch.tensor([dimer_id], dtype=torch.long)
    
    data = Data(
        ZA=ZA,
        RA=RA,
        ZB=ZB,
        RB=RB,
        e_ABsr_source=e_ABsr_source,
        e_ABsr_target=e_ABsr_target,
        e_ABlr_source=e_ABlr_source,
        e_ABlr_target=e_ABlr_target,
        dimer_ind=dimer_ind,
        dimer_ind_lr=dimer_ind.clone(),
        e_AA_source=e_AA_source,
        e_AA_target=e_AA_target,
        e_BB_source=e_BB_source,
        e_BB_target=e_BB_target,
        molecule_ind_A=torch.zeros(natoms_A, dtype=torch.long),
        molecule_ind_B=torch.zeros(natoms_B, dtype=torch.long),
        total_charge_A=torch.tensor([0.0]),
        total_charge_B=torch.tensor([0.0]),
        qA=torch.randn(natoms_A, 1),
        muA=torch.randn(natoms_A, 3),
        quadA=torch.randn(natoms_A, 5),
        hlistA=torch.randn(natoms_A, 3),
        qB=torch.randn(natoms_B, 1),
        muB=torch.randn(natoms_B, 3),
        quadB=torch.randn(natoms_B, 5),
        hlistB=torch.randn(natoms_B, 3),
    )
    return data


def test_ap3_collate_concatenation_order():
    data1 = create_simple_test_data(natoms_A=3, natoms_B=3, n_sr_edges=5, n_lr_edges=3, dimer_id=0)
    data2 = create_simple_test_data(natoms_A=4, natoms_B=2, n_sr_edges=4, n_lr_edges=2, dimer_id=1)
    
    batch = ap3_fused_collate_update_no_target([data1, data2])
    
    n_sr = batch.e_ABsr_source.size(0)
    n_lr = batch.e_ABlr_source.size(0)
    n_full = batch.e_ABfull_source.size(0)
    
    assert n_full == n_sr + n_lr, f"Full size {n_full} != SR size {n_sr} + LR size {n_lr}"
    
    assert torch.allclose(batch.e_ABfull_source[:n_sr], batch.e_ABsr_source)
    assert torch.allclose(batch.e_ABfull_source[n_sr:], batch.e_ABlr_source)
    assert torch.allclose(batch.e_ABfull_target[:n_sr], batch.e_ABsr_target)
    assert torch.allclose(batch.e_ABfull_target[n_sr:], batch.e_ABlr_target)
    
    n_dimer_sr = batch.dimer_ind.size(0)
    n_dimer_lr = batch.dimer_ind_lr.size(0)
    n_dimer_full = batch.dimer_ind_full.size(0)
    
    assert n_dimer_full == n_dimer_sr + n_dimer_lr
    assert torch.allclose(batch.dimer_ind_full[:n_dimer_sr], batch.dimer_ind)
    assert torch.allclose(batch.dimer_ind_full[n_dimer_sr:], batch.dimer_ind_lr)


def test_ap3_collate_edge_offsets_single_batch():
    data = create_simple_test_data(natoms_A=3, natoms_B=3, n_sr_edges=5, n_lr_edges=3, dimer_id=0)
    
    batch = ap3_fused_collate_update_no_target([data])
    
    assert torch.allclose(batch.e_ABsr_source, data.e_ABsr_source)
    assert torch.allclose(batch.e_ABsr_target, data.e_ABsr_target)
    assert torch.allclose(batch.e_ABlr_source, data.e_ABlr_source)
    assert torch.allclose(batch.e_ABlr_target, data.e_ABlr_target)
    
    manual_full_source = torch.cat([data.e_ABsr_source, data.e_ABlr_source], dim=0)
    manual_full_target = torch.cat([data.e_ABsr_target, data.e_ABlr_target], dim=0)
    
    assert torch.allclose(batch.e_ABfull_source, manual_full_source)
    assert torch.allclose(batch.e_ABfull_target, manual_full_target)


def test_ap3_collate_edge_offsets_multi_batch():
    data1 = create_simple_test_data(natoms_A=3, natoms_B=3, n_sr_edges=5, n_lr_edges=3, dimer_id=0)
    data2 = create_simple_test_data(natoms_A=4, natoms_B=2, n_sr_edges=4, n_lr_edges=2, dimer_id=1)
    
    batch = ap3_fused_collate_update_no_target([data1, data2])
    
    natoms_A1 = data1.RA.size(0)
    natoms_B1 = data1.RB.size(0)
    natoms_A2 = data2.RA.size(0)
    natoms_B2 = data2.RB.size(0)
    
    n_sr_1 = data1.e_ABsr_source.size(0)
    n_lr_1 = data1.e_ABlr_source.size(0)
    n_sr_2 = data2.e_ABsr_source.size(0)
    n_lr_2 = data2.e_ABlr_source.size(0)
    
    expected_e_ABsr_source_1 = data1.e_ABsr_source
    expected_e_ABsr_target_1 = data1.e_ABsr_target
    expected_e_ABsr_source_2 = data2.e_ABsr_source + natoms_A1
    expected_e_ABsr_target_2 = data2.e_ABsr_target + natoms_B1
    
    assert torch.allclose(batch.e_ABsr_source[:n_sr_1], expected_e_ABsr_source_1)
    assert torch.allclose(batch.e_ABsr_target[:n_sr_1], expected_e_ABsr_target_1)
    assert torch.allclose(batch.e_ABsr_source[n_sr_1:], expected_e_ABsr_source_2)
    assert torch.allclose(batch.e_ABsr_target[n_sr_1:], expected_e_ABsr_target_2)
    
    expected_e_ABlr_source_1 = data1.e_ABlr_source
    expected_e_ABlr_target_1 = data1.e_ABlr_target
    expected_e_ABlr_source_2 = data2.e_ABlr_source + natoms_A1
    expected_e_ABlr_target_2 = data2.e_ABlr_target + natoms_B1
    
    assert torch.allclose(batch.e_ABlr_source[:n_lr_1], expected_e_ABlr_source_1)
    assert torch.allclose(batch.e_ABlr_target[:n_lr_1], expected_e_ABlr_target_1)
    assert torch.allclose(batch.e_ABlr_source[n_lr_1:], expected_e_ABlr_source_2)
    assert torch.allclose(batch.e_ABlr_target[n_lr_1:], expected_e_ABlr_target_2)
    
    expected_full_source = torch.cat([
        expected_e_ABsr_source_1,
        expected_e_ABsr_source_2,
        expected_e_ABlr_source_1,
        expected_e_ABlr_source_2
    ], dim=0)
    expected_full_target = torch.cat([
        expected_e_ABsr_target_1,
        expected_e_ABsr_target_2,
        expected_e_ABlr_target_1,
        expected_e_ABlr_target_2
    ], dim=0)
    
    assert torch.allclose(batch.e_ABfull_source, expected_full_source)
    assert torch.allclose(batch.e_ABfull_target, expected_full_target)


def test_ap3_collate_with_targets():
    data1 = create_simple_test_data(natoms_A=3, natoms_B=3, n_sr_edges=5, n_lr_edges=3, dimer_id=0)
    data2 = create_simple_test_data(natoms_A=4, natoms_B=2, n_sr_edges=4, n_lr_edges=2, dimer_id=1)
    
    data1.E_tot = torch.tensor([1.0])
    data1.E_tot_mon_A = torch.tensor([0.5])
    data1.E_tot_mon_B = torch.tensor([0.5])
    data1.E_int = torch.tensor([0.0])
    
    data2.E_tot = torch.tensor([2.0])
    data2.E_tot_mon_A = torch.tensor([1.0])
    data2.E_tot_mon_B = torch.tensor([1.0])
    data2.E_int = torch.tensor([0.0])
    
    batch = ap3_fused_collate_update([data1, data2])
    
    assert hasattr(batch, 'E_tot')
    assert hasattr(batch, 'E_int')
    assert hasattr(batch, 'e_ABfull_source')
    assert hasattr(batch, 'e_ABfull_target')
    assert hasattr(batch, 'dimer_ind_full')
    
    assert torch.allclose(batch.E_tot, torch.tensor([1.0, 2.0]))


def test_ap3_collate_monomer_indices():
    data1 = create_simple_test_data(natoms_A=3, natoms_B=3, n_sr_edges=5, n_lr_edges=3, dimer_id=0)
    data2 = create_simple_test_data(natoms_A=4, natoms_B=2, n_sr_edges=4, n_lr_edges=2, dimer_id=1)
    
    batch = ap3_fused_collate_update_no_target_monomer_indices([data1, data2])
    
    assert torch.all(batch.dimer_ind == 0) or torch.all(batch.dimer_ind == 1) or (
        torch.any(batch.dimer_ind == 0) and torch.any(batch.dimer_ind == 1)
    )
    
    assert torch.all(batch.dimer_ind_lr == 0) or torch.all(batch.dimer_ind_lr == 1) or (
        torch.any(batch.dimer_ind_lr == 0) and torch.any(batch.dimer_ind_lr == 1)
    )
    
    assert hasattr(batch, 'e_ABfull_source')
    assert hasattr(batch, 'e_ABfull_target')
    assert hasattr(batch, 'dimer_ind_full')


def test_ap3_collate_preserve_original_fields():
    data = create_simple_test_data(natoms_A=3, natoms_B=3, n_sr_edges=5, n_lr_edges=3, dimer_id=0)
    
    batch = ap3_fused_collate_update_no_target([data])
    
    assert hasattr(batch, 'e_ABsr_source')
    assert hasattr(batch, 'e_ABsr_target')
    assert hasattr(batch, 'e_ABlr_source')
    assert hasattr(batch, 'e_ABlr_target')
    assert hasattr(batch, 'dimer_ind')
    assert hasattr(batch, 'dimer_ind_lr')
    
    assert hasattr(batch, 'e_ABfull_source')
    assert hasattr(batch, 'e_ABfull_target')
    assert hasattr(batch, 'dimer_ind_full')


if __name__ == "__main__":
    test_ap3_collate_concatenation_order()
    test_ap3_collate_edge_offsets_single_batch()
    test_ap3_collate_edge_offsets_multi_batch()
    test_ap3_collate_with_targets()
    test_ap3_collate_monomer_indices()
    test_ap3_collate_preserve_original_fields()
    print("All tests passed!")
