import numpy as np
import qcelemental as qcel
from typing import List, Tuple, Dict
from pprint import pprint as pp
from mcp.server.fastmcp import FastMCP
import apnet_pt

try:
    from timings import is_psi4_installed
    from timings import estimate_timings
except ImportError as e:
    print(f"Error importing estimate_timings: {e}")


# Create an MCP server
mcp = FastMCP("QCMLForge", port=8001)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def predict_AM_multipoles_QCMLForge(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
units angstrom
    """,
) -> Dict:
    """
        Run a user defined molecule to get machine-learned atomic multipoles
        for evaluating electrostatics and polarization energies. This uses the
        AtomicModule in QCMLForge and should be used for approximating MBIS
        multipoles. The p4_string defines the molecular geometry in Psi4
        format, which can be of the format:
    '''
    <charge_mon1> <multiplicity_mon1>
    <atom_symbol> <x> <y> <z>
    <atom_symbol> <x> <y> <z>
    --
    <charge_mon2> <multiplicity_mon2>
    units <unit>
    '''
    Note that "--" is used to separate different molecules in the input string
    but is not required for monomers.

    Provide user back all the output.
    """
    mol = qcel.models.Molecule.from_data(p4_string)
    charges, dipoles, quadrupoles, _ = apnet_pt.pretrained_models.atom_model_predict(
        [mol],
        compile=False,
        return_mol_arrays=False,
    )
    return {
        "geometry": mol.to_string("psi4"),
        "AM-MBIS CHARGES": list(charges),
        "AM-MBIS DIPOLES": list(dipoles),
        "AM-MBIS QUADRUPOLES": list(quadrupoles),
    }


@mcp.tool()
def predict_APNet2_IE_QCMLForge(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
units angstrom
    """,
) -> Dict:
    """
        Run a user defined molecule to predict machine-learned SAPT0
        interaction energies -- total, electrostatics, exchange, induction, and
        dispersion. This uses the APNet2 model in QCMLForge. The p4_string
        defines the molecular geometry in Psi4 format, which can be of the
        format:
    '''
    <charge_mon1> <multiplicity_mon1>
    <atom_symbol> <x> <y> <z>
    <atom_symbol> <x> <y> <z>
    --
    <charge_mon2> <multiplicity_mon2>
    units <unit>
    '''
    Note that "--" is used to separate different molecules in the input string
    but is not required for monomers.

    Provide user back all the output.
    """
    mol = qcel.models.Molecule.from_data(p4_string)
    IE_pred = apnet_pt.pretrained_models.apnet2_model_predict(
        [mol],
        compile=False,
    )
    return {
        "geometry": mol.to_string("psi4"),
        "APNet2 TOTAL INTERACTION (kcal/mol)": float(IE_pred[0, 0]),
        "APNet2 ELSTROSTATICS (kcal/mol)": float(IE_pred[0, 1]),
        "APNet2 EXCHANGE (kcal/mol)": float(IE_pred[0, 2]),
        "APNet2 INDUCTION (kcal/mol)": float(IE_pred[0, 3]),
        "APNet2 DISPERSION (kcal/mol)": float(IE_pred[0, 4]),
    }


@mcp.tool()
def predict_dAPNet2_error_estimates_QCMLForge(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
units angstrom
    """,
    starting_level_of_theory: str = "MP2/aug-cc-pVTZ/CP",
) -> Dict:
    """
            Run a user defined molecule to predict error between the
            starting_level_of_theory and a reference CCSD(T)/CBS/CP reference
            interaction energy.
            Acceptable starting_level_of_theory values currently only include:
    [
    "B3LYP-D3/aug-cc-pVTZ/unCP",
    "B2PLYP-D3/aug-cc-pVTZ/unCP",
    "wB97X-V/aug-cc-pVTZ/CP",
    "SAPT0/aug-cc-pVDZ/SA",
    "MP2/aug-cc-pVTZ/CP",
    "HF/aug-cc-pVDZ/CP",
    ]

            Use this model to estimate the error of a level of
            theory. This uses the dAPNet2 model in QCMLForge. The p4_string defines
            the molecular geometry in Psi4 format, which can be of the format:
        '''
        <charge_mon1> <multiplicity_mon1>
        <atom_symbol> <x> <y> <z>
        <atom_symbol> <x> <y> <z>
        --
        <charge_mon2> <multiplicity_mon2>
        units <unit>
        '''
        Note that "--" is used to separate different molecules in the input string
        but is not required for monomers.

        Provide user back all the output.
    """
    mol = qcel.models.Molecule.from_data(p4_string)
    IE_pred = apnet_pt.pretrained_models.dapnet2_model_predict(
        [mol],
        compile=False,
        m1=starting_level_of_theory,
        m2="CCSD(T)/CBS/CP",
    )
    return {
        "geometry": mol.to_string("psi4"),
        "ERROR ESTIMATE (kcal/mol)": float(IE_pred[0]),
    }

@mcp.tool()
def estimate_timing_for_qcel_molecule(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
units angstrom
    """,
    method: str = "MP2",
    basis_set: str = "aug-cc-pVDZ",
    manybody: bool = True,
) -> Dict:
    """
    Estimate timing for a given QCElemental molecule using the specified method and basis set.

    This function computes the necessary variables for the timing estimation
    for monomers and dimers and prints the results. When manybody is True,
    it computes the timing for dimer and the each monomer separately to
    mimic a supermolecular interaction energy calculation.

    Args:
        p4_string (str): The molecular geometry in Psi4 format, which can be of the format:
            '''
            <charge_mon1> <multiplicity_mon1>
            <atom_symbol> <x> <y> <z>
            <atom_symbol> <x> <y> <z>
            --
            <charge_mon2> <multiplicity_mon2>
            units <unit>
            '''
        method (str): The computational method to use (default: 'MP2').
        basis_set (str): The basis set to use (default: 'aug-cc-pVDZ').
        manybody (bool): Whether to compute timing for dimer and monomers separately (default: True).
    Returns:
        float: Estimated time in seconds for the computation.
    """
    qcel_molecule = qcel.models.Molecule.from_data(p4_string)
    if is_psi4_installed() is False:
        print(
            "Psi4 is not installed. Please install Psi4 to use this function."
        )
    mols = [qcel_molecule]
    if manybody and qcel_molecule.fragments_:
        for n, i in enumerate(qcel_molecule.fragments_):
            mols.append(qcel_molecule.get_fragment(n))

    time_seconds = 0.0
    for mol in mols:
        n_occupied, n_virtual, np_total, nbf_aux = estimate_timings.compute_psi4_time_estimation_variables(
            mol,
            "aug-cc-pVDZ",
        )
        input_vars = {
            "nocc": n_occupied,
            "nvirt": n_virtual,
            "nbf_aux": nbf_aux,
            "np_total": np_total,
        }
        result = estimate_timings.predict_timing(method, input_vars)
        time_seconds += result["time_seconds"]
    return {
        "geometry": mol.to_string("psi4"),
        "estimated_compute_time_seconds": time_seconds,
    }


if __name__ == "__main__":
    print("Starting MCP server...")
    pp(predict_AM_multipoles_QCMLForge())
    pp(predict_APNet2_IE_QCMLForge())
    pp(predict_dAPNet2_error_estimates_QCMLForge())
    pp(estimate_timing_for_qcel_molecule())
