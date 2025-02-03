# from . import apnet2_model
from . import atom_model
from . import atomic_datasets
from qcelemental.models.molecule import Molecule
import os
import numpy as np

model_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../models/"


def atom_model_predict(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 3,
):
    num_models = 5
    am = atom_model.AtomModel(
        pre_trained_model_path=f"{model_dir}am_ensemble/am_0.pt",
    )
    if compile:
        print("Compiling models...")
        am.compile_model()
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
    for batch in batched_data:
        # Intermediates which get averaged from num_models
        qs_t = np.zeros((len(batch.x)))
        ds_t = np.zeros((len(batch.x), 3))
        qps_t = np.zeros((len(batch.x), 3, 3))
        for i in range(num_models):
            q, d, qp, _ = am.predict_multipoles_batch(
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
        atom_idx += len(batch.x)
    return pred_qs, pred_ds, pred_qps
