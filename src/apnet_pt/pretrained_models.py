from . import AtomPairwiseModels
from . import AtomModels
from . import atomic_datasets
from qcelemental.models.molecule import Molecule
import os
import numpy as np
import copy
from importlib import resources
from apnet_pt.pt_datasets.dapnet_ds import clean_str_for_filename

# model_dir = os.path.dirname(os.path.realpath(__file__)) + "/models/"
model_dir = resources.files("apnet_pt").joinpath("models")


def atom_model_predict(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 3,
    return_mol_arrays: bool = True,
):
    num_models = 5
    am = AtomModels.ap2_atom_model.AtomModel(
        # pre_trained_model_path=f"{model_dir}am_ensemble/am_0.pt",
        pre_trained_model_path=resources.files("apnet_pt").joinpath(
            "models", "am_ensemble", "am_0.pt"
        ),
    )
    if compile:
        print("Compiling models...")
        am.compile_model()
    models = [copy.deepcopy(am) for _ in range(num_models)]
    for i in range(1, num_models):
        models[i].set_pretrained_model(
            # model_path=f"{model_dir}am_ensemble/am_{i}.pt",
            model_path=resources.files("apnet_pt").joinpath(
                "models", "am_ensemble", f"am_{i}.pt"
            ),
        )
    print("Processing mols...")
    data = [
        atomic_datasets.qcel_mon_to_pyg_data(mol, r_cut=am.model.r_cut) for mol in mols
    ]
    print(data)
    batched_data = [
        atomic_datasets.atomic_collate_update_no_target(
            data[i: i + batch_size])
        for i in range(0, len(data), batch_size)
    ]
    print(f"Number of batches: {len(batched_data)}")
    print(f"{batched_data = }")
    print(f"{batched_data[0].x = }")
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
                batch,
                isolate_predictions=False,
            )
            qs_t += q.numpy()
            ds_t += d.numpy()
            qps_t += qp.numpy()
        qs_t /= num_models
        ds_t /= num_models
        qps_t /= num_models
        pred_qs[atom_idx: atom_idx + len(batch.x)] = qs_t
        pred_ds[atom_idx: atom_idx + len(batch.x)] = ds_t
        pred_qps[atom_idx: atom_idx + len(batch.x)] = qps_t
        unique_values, repeats = np.unique(
            [batch.molecule_ind[i] for i in range(len(batch.molecule_ind))],
            return_counts=True,
        )
        mol_id_ranges = []
        for i in repeats:
            mol_id_ranges.append(int(i) + atom_idx)
            atom_idx += int(i)
        mol_ids.extend(mol_id_ranges)
    if return_mol_arrays:
        # Drop the final mol_ids because the split will take the full last
        # slice
        pred_qs = np.split(pred_qs, mol_ids[:-1])
        pred_ds = np.split(pred_ds, mol_ids[:-1])
        pred_qps = np.split(pred_qps, mol_ids[:-1])
        return pred_qs, pred_ds, pred_qps
    return pred_qs, pred_ds, pred_qps, mol_ids


def apnet2_model_predict(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 16,
    ensemble_model_dir: str = model_dir,
):
    num_models = 5
    ap2 = AtomPairwiseModels.apnet2.APNet2Model(
        # pre_trained_model_path=f"{ensemble_model_dir}ap2_ensemble/ap2_0.pt",
        # atom_model_pre_trained_path=f"{ensemble_model_dir}am_ensemble/am_0.pt",
        pre_trained_model_path=resources.files("apnet_pt").joinpath(
            "models", "ap2_ensemble", "ap2_0.pt"
        ),
        atom_model_pre_trained_path=resources.files("apnet_pt").joinpath(
            "models", "am_ensemble", "am_0.pt"
        ),
    )
    if compile:
        print("Compiling models...")
        ap2.compile_model()
    models = [copy.deepcopy(ap2) for _ in range(num_models)]
    for i in range(1, num_models):
        models[i].set_pretrained_model(
            # ap2_model_path=f"{ensemble_model_dir}ap2_ensemble/ap2_{i}.pt",
            # am_model_path=f"{ensemble_model_dir}am_ensemble/am_{i}.pt",
            ap2_model_path=resources.files("apnet_pt").joinpath(
                "models", "ap2_ensemble", f"ap2_{i}.pt"
            ),
            am_model_path=resources.files("apnet_pt").joinpath(
                "models", "am_ensemble", f"am_{i}.pt"
            ),
        )
    pred_IEs = np.zeros((len(mols), 5))
    print("Processing mols...")
    for i in range(num_models):
        IEs = models[i].predict_qcel_mols(mols, batch_size=batch_size)
        pred_IEs[:, 1:] += IEs
        pred_IEs[:, 0] += np.sum(IEs, axis=1)
    pred_IEs /= num_models
    return pred_IEs


def dapnet2_levels_of_theory_pretrained():
    """
    Returns a list of possible levels of theory with pretrained models to use
    in the dapnet2_model_predict(). Note that these pretrained models
    predicts E=(m1-CCSD(T)/CBS/CP)
    """
    return [
        "B3LYP-D3/aug-cc-pVTZ/unCP",
        "B2PLYP-D3/aug-cc-pVTZ/unCP",
        "wB97X-V/aug-cc-pVTZ/CP",
        "SAPT0/aug-cc-pVDZ/SA",
        "MP2/aug-cc-pVTZ/CP",
        "HF/aug-cc-pVDZ/CP",
    ]


def dapnet2_model_predict(
    mols: [Molecule],
    m1: str,
    m2: str,
    compile: bool = True,
    pre_trained_model_path: str = None,
    batch_size: int = 16,
    use_GPU: bool = None,
) -> np.ndarray:
    atom_model = AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=use_GPU,
    ).set_pretrained_model(model_id=0)
    apnet2 = AtomPairwiseModels.apnet2.APNet2Model(
        atom_model=atom_model.model,
        use_GPU=use_GPU,
    ).set_pretrained_model(model_id=0)
    apnet2.model.return_hidden_states = True
    if pre_trained_model_path is None:
        assert m1 in dapnet2_levels_of_theory_pretrained(), (
            f"Pretrained model for {m1} not found. "
            f"Please use one of the following: {dapnet2_levels_of_theory_pretrained()}"
        )
        assert m2 == "CCSD(T)/CBS/CP", (
            "Pretrained models only predict m2=CCSD(T)/CBS/CP"
        )
        pre_trained_model_path = resources.files("apnet_pt").joinpath(
            "models", "dapnet2", f"{clean_str_for_filename(m1)}_to_{clean_str_for_filename(m2)}_0.pt"
        )
    dapnet2 = AtomPairwiseModels.dapnet2.dAPNet2Model(
        atom_model=atom_model,
        apnet2_model=apnet2,
        pre_trained_model_path=pre_trained_model_path,
        use_GPU=use_GPU,
    )
    return dapnet2.predict_qcel_mols(mols, batch_size=batch_size)
