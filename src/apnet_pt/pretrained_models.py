from . import AtomPairwiseModels
from . import AtomModels
from . import atomic_datasets
from qcelemental.models.molecule import Molecule
import os
import numpy as np
import copy

model_dir = os.path.dirname(os.path.realpath(__file__)) + "/models/"


def atom_model_predict(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 3,
    return_mol_arrays: bool = True,
):
    num_models = 5
    am = AtomModels.ap2_atom_model.AtomModel(
        pre_trained_model_path=f"{model_dir}am_ensemble/am_0.pt",
    )
    if compile:
        print("Compiling models...")
        am.compile_model()
    models = [copy.deepcopy(am) for _ in range(num_models)]
    for i in range(1, num_models):
        models[i].set_pretrained_model(
            model_path=f"{model_dir}am_ensemble/am_{i}.pt",
        )
    print("Processing mols...")
    data = [atomic_datasets.qcel_mon_to_pyg_data(
        mol, r_cut=am.model.r_cut) for mol in mols]
    batched_data = [
        atomic_datasets.atomic_collate_update_no_target(data[i:i + batch_size])
        for i in range(0, len(data), batch_size)
    ]
    print("Predicting...")
    atom_count = sum([len(d.x) for d in data])
    pred_qs = np.zeros((atom_count))
    pred_ds = np.zeros((atom_count, 3))
    pred_qps = np.zeros((atom_count, 3, 3))
    atom_idx = 0
    mol_ids = []
    for batch in batched_data:
        # Intermediates which get averaged from num_models
        qs_t = np.zeros((len(batch.x)))
        ds_t = np.zeros((len(batch.x), 3))
        qps_t = np.zeros((len(batch.x), 3, 3))
        for i in range(num_models):
            q, d, qp, _ = models[i].predict_multipoles_batch(
                batch, isolate_predictions=False,
            )
            qs_t += q.numpy()
            ds_t += d.numpy()
            qps_t += qp.numpy()
        qs_t /= num_models
        ds_t /= num_models
        qps_t /= num_models
        pred_qs[atom_idx:atom_idx + len(batch.x)] = qs_t
        pred_ds[atom_idx:atom_idx + len(batch.x)] = ds_t
        pred_qps[atom_idx:atom_idx + len(batch.x)] = qps_t
        tmp = np.unique([batch.molecule_ind[i] for i in range(len(batch.molecule_ind))], return_counts=True)
        mol_id_ranges = [atom_idx]
        for i in range(len(tmp[1]) - 1):
            mol_id_ranges.append(int(mol_id_ranges[i] + tmp[1][i + 1]))
        atom_idx += len(batch.x)
        mol_ids.extend(
            mol_id_ranges
        )
    mol_ids.append(atom_idx)
    if return_mol_arrays:
        pred_qs = np.split(pred_qs, mol_ids[1:-1])
        pred_ds = np.split(pred_ds, mol_ids[1:-1])
        pred_qps = np.split(pred_qps, mol_ids[1:-1])
        return pred_qs, pred_ds, pred_qps
    return pred_qs, pred_ds, pred_qps, mol_ids


def apnet2_model_predict(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 16,
):
    num_models = 5
    ap2 = AtomPairwiseModels.apnet2.APNet2Model(
        pre_trained_model_path=f"{model_dir}ap2_ensemble/ap2_0.pt",
        atom_model_pre_trained_path=f"{model_dir}am_ensemble/am_0.pt",
    )
    if compile:
        print("Compiling models...")
        ap2.compile_model()
    models = [copy.deepcopy(ap2) for _ in range(num_models)]
    for i in range(1, num_models):
        models[i].set_pretrained_model(
            ap2_model_path=f"{model_dir}ap2_ensemble/ap2_{i}.pt",
            am_model_path=f"{model_dir}am_ensemble/am_{i}.pt",
        )
    pred_IEs = np.zeros((len(mols), 5))
    print("Processing mols...")
    for i in range(num_models):
        IEs = models[i].predict_qcel_mols(mols, batch_size=batch_size)
        pred_IEs[:, 1:] += IEs
        pred_IEs[:, 0] += np.sum(IEs, axis=1)
    pred_IEs /= num_models
    return pred_IEs
