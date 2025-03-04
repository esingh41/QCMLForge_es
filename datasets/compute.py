import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from qcportal import PortalClient

# from qcportal.manybody import ManybodyDataset, ManybodyDatasetEntry, ManybodySpecification, ManybodyKeywords, BSSECorrectionEnum
from qcportal.singlepoint import SinglepointDataset, SinglepointDatasetEntry
from qcportal.singlepoint import QCSpecification
from qcelemental.models import Molecule
from pprint import pprint as pp
from qm_tools_aw import tools
import qcelemental as qcel

client = PortalClient("http://localhost:7777", verify=False)
print(client)
ROOT = os.getcwd()

# df['qcel'] are qcelemental monomer Molecule objects
# df = pd.read_pickle("./data_dir/raw/monomers_ap3_spec_1.pkl")
# df.head().to_pickle("./data_dir/raw/monomers_ap3_spec_1_head.pkl")
df = pd.read_pickle("./data_dir/raw/monomers_ap3_spec_1_head.pkl")
print(df.columns.values)


def add_QCdataset(
    df: pd.DataFrame = df,
    system: str = "AP2-monomers",
    method: str = "PBE0",
    basis: str = "aug-cc-pVTZ",
    mode: str = "create",
) -> SinglepointDataset:
    """
    Add a dataset with molecules and specification to client

    Must be in a directory with input files to read from (at least to get Molecule entries)
    """
    # client.delete_dataset(dataset_id=1, delete_records=True)
    try:
        ds = client.add_dataset("singlepoint", system,
                                f"Dataset to contain {system}")
        print(f"Added {system} as dataset")
    except Exception:
        ds = client.get_dataset("singlepoint", system)
        print(f"Found {system} dataset, using this instead")
        print(ds)

    if mode == "create":
        entry_list = []
        for idx, row in df.iterrows():
            name = row["name"]
            extras = {
                "name": name,
                "idx": idx,
            }
            mol = row["qcel"]
            mol = Molecule.from_data(mol.dict(), extras=extras)
            ent = SinglepointDatasetEntry(name=name, molecule=mol)
            entry_list.append(ent)
            break

        ds.add_entries(entry_list)
        print("Added molecules to dataset")

        spec = QCSpecification(
            program="psi4",
            driver="energy",
            method=method,
            basis=basis,
            keywords={
                "d_convergence": 8,
                "dft_radial_points": 99,
                "dft_spherical_points": 590,
                "e_convergence": 10,
                "guess": "sad",
                "mbis_d_convergence": 9,
                "mbis_maxiter": 1000,
                "mbis_radial_points": 99,
                "mbis_spherical_points": 590,
                "scf_properties": ["mbis_charges"],
                "scf_type": "df",
            },
            protocols={"wavefunction": "orbitals_and_eigenvalues"},
            extras={"description": "MBIS calculations for Hirshfeld Volumes"}
        )
        ds.add_specification(name=f"psi4/{method}/{basis}", specification=spec)
        print(f"Added {method}/{basis} specification to dataset")
    return ds


def submit(ds: SinglepointDataset, tag: str = "short") -> None:
    """
    Submit a dataset for computation
    ** Not super useful atm, just calls submit on dataset
    """
    ds.submit(tag=tag)  # will default to submit all specifications for all entries
    print(f"Submitted {ds.name} dataset")


def main():
    mode = "analyze"
    system = "AP2-monomers"
    method = "PBE0"
    basis = "aug-cc-pVTZ"
    ds = add_QCdataset(df, system, method=method, basis=basis, mode=mode)
    if mode == "create" or mode == "run":
        if mode == "run":
            submit(ds, tag="normal")
    else:
        ds.status()
        # return
        for entry_name, spec_name, record in tqdm(
            ds.iterate_records(status="complete")
        ):
            record_dict = record.dict()
            qcvars = record_dict["properties"]
            charges = qcvars["mbis charges"]
            dipoles = qcvars["mbis dipoles"]
            quadrupoles = qcvars["mbis quadrupoles"]
            level_of_theory = f"{record_dict['specification']['method']}/{record_dict['specification']['basis']}"
            pp(record_dict)
            pp(qcvars)
            break
        df2 = ds.get_properties_df(['mbis charges', 'mbis dipoles', 'mbis quadrupoles', 'mbis radial moments <r^2>', 'mbis radial moments <r^3>', 'mbis radial moments <r^4>', 'mbis valence widths'])
        print(df2)


if __name__ == "__main__":
    main()
