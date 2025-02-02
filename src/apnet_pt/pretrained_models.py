from . import apnet2_model
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
        print("Compiling models")
        am.compile_model()
    preds = [[] for i in range(num_models)]
    print("Processing mols...")
    data = [atomic_datasets.qcel_mon_to_pyg_data(mol, r_cut=am.model.r_cut) for mol in mols]
    batched_data = [
        atomic_datasets.atomic_collate_update_no_target(data[i:i + batch_size])
        for i in range(0, len(data), batch_size)
    ]
    print("Predicting...")
    for i in range(num_models):
        for batch in batched_data:
            output = am.predict_multipoles_batch(batch)
            preds[i].extend(output)
    # TODO: need to stack the predictions
    return preds
