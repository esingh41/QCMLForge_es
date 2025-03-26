import os
import argparse
import pandas as pd
from tqdm import tqdm
from qcportal import PortalClient
import numpy as np

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
# df.head().to_pickle("./data_dir/raw/monomers_ap3_spec_1_head.pkl")
# df = pd.read_pickle("../data_dir/raw/monomers_ap3_spec_1_head.pkl")
# print(df.columns.values)


def add_QCdataset(
    df: pd.DataFrame,
    system: str = "AP2-monomers",
    method: str = "PBE0",
    basis: str = "aug-cc-pVTZ",
    mode: str = "create",
) -> SinglepointDataset:
    """
    Add a dataset with molecules and specification to client

    Must be in a directory with input files to read from (at least to get Molecule entries)
    """
    # client.delete_dataset(dataset_id=2, delete_records=True)
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

        ds.add_entries(entry_list)
        print(f"Added {len(entry_list)} molecules to dataset")

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
                "scf_properties": ["mbis_charges", "MBIS_VOLUME_RATIOS"],
                "scf_type": "df",
            },
            protocols={"wavefunction": "orbitals_and_eigenvalues"},
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
    # mode = "create"
    # mode = "run"
    mode = "progess"
    mode = "analyze"
    system = "AP2-monomers"
    method = "PBE0"
    basis = "aug-cc-pVTZ"

    print(f"Running in {mode} mode")

    df = pd.read_pickle("../data_dir/raw/monomers_ap3_spec_1.pkl")
    ds = add_QCdataset(df, system, method=method, basis=basis, mode=mode)
    if mode == "create" or mode == "run":
        if mode == "run":
            submit(ds, tag="normal")
    elif mode == "progess":
        ds.status()
        ds.print_status()
    else:
        ds.status()
        ds.print_status()
        # return
        cnt = 0
        data = {
            "id": [],
            "name": [],
            "Z": [],
            "R": [],
            "cartesian_multipoles": [],
            "entry_name": [],
            "spec_name": [],
            "TQ": [],
            "molecular_multiplicity": [],
            "volume ratios": [],
            "valence widths": [],
            "radial moments <r^2>": [],
            "radial moments <r^3>": [],
            "radial moments <r^4>": [],
        }
        for entry_name, spec_name, record in tqdm(
            ds.iterate_records(status="complete")
        ):
            record_dict = record.dict()
            qcvars = record_dict["properties"]
            charges = qcvars["mbis charges"]
            dipoles = qcvars["mbis dipoles"]
            quadrupoles = qcvars["mbis quadrupoles"]
            level_of_theory = f"{record_dict['specification']['method']}/{record_dict['specification']['basis']}"

            n = len(charges)

            charges = np.reshape(charges, (n, 1))
            dipoles = np.reshape(dipoles, (n, 3))
            quad = np.reshape(quadrupoles, (n, 3, 3))

            quad = [q[np.triu_indices(3)] for q in quad]
            quadrupoles = np.array(quad)
            multipoles = np.concatenate(
                [charges, dipoles, quadrupoles], axis=1)

            data['volume ratios'].append(qcvars['mbis volume ratios'])
            data['valence widths'].append(qcvars['mbis valence widths'])
            data['radial moments <r^2>'].append(qcvars['mbis radial moments <r^2>'])
            data['radial moments <r^3>'].append(qcvars['mbis radial moments <r^3>'])
            data['radial moments <r^4>'].append(qcvars['mbis radial moments <r^4>'])
            data["id"].append(record.molecule.extras['idx'])
            data["name"].append(record.molecule.extras['name'])
            data["Z"].append(record.molecule.atomic_numbers)
            data["R"].append(record.molecule.geometry * qcel.constants.bohr2angstroms)
            data["cartesian_multipoles"].append(multipoles)
            data["entry_name"].append(entry_name)
            data["spec_name"].append(spec_name)
            data["TQ"].append(int(record.molecule.molecular_charge))
            data["molecular_multiplicity"].append(
                record.molecule.molecular_multiplicity
            )
            # print(qcvars.keys())
            cnt += 1
        df2 = pd.DataFrame(data)
        print(df2)
        df2.to_pickle(f"../data_dir/raw/monomers_ap3_spec_1_pbe0.pkl")
            # break
        # df2 = ds.get_properties_df(['mbis charges', 'mbis dipoles', 'mbis quadrupoles', 'mbis radial moments <r^2>', 'mbis radial moments <r^3>', 'mbis radial moments <r^4>', 'mbis valence widths', 'mbis volume ratios', 'mbis free atom c volume', 'mbis free atom h volume', 'mbis free atom n volume', 'mbis free atom s volume',])
        # df2 = ds.get_properties_df(['molecule_id'])
        print(df2)
        print(df2.columns.values)


if __name__ == "__main__":
    main()
