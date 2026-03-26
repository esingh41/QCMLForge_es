"""Coverage for model_print support across major model families."""

import io
from typing import cast

import pytest
from torch.nn.parameter import UninitializedParameter

from apnet_pt import get_model_info, print_model_tree
from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN, AtomModel
from apnet_pt.AtomModels.ap2_hirshfeld_atom_model import AtomHirshfeldMPNN
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2_MPNN, APNet2Model
from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_Model
from apnet_pt.AtomPairwiseModels.dapnet2 import dAPNet2Model
from apnet_pt.AtomPairwiseModels.mtp_mtp import AtomTypeParamNN, DimerProp


def count_params(module, trainable_only=False):
    """Count parameters, treating lazy uninitialized tensors as size zero."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(
        0 if isinstance(p, UninitializedParameter) else p.numel() for p in params
    )


def build_demo_atom_model():
    return AtomModel(
        use_GPU=False,
        ignore_database_null=True,
        # n_message=1,
        # n_rbf=4,
        # n_neuron=16,
        # n_embed=4,
    )


def build_demo_apnet2_model():
    atom_model = AtomMPNN(
            # n_message=1, n_rbf=4, n_neuron=16, n_embed=4
            )
    return APNet2Model(
        atom_model=atom_model,
        use_GPU=False,
        ignore_database_null=True,
        # n_message=1,
        # n_rbf=4,
        # n_neuron=16,
        # n_embed=4,
    )


def build_demo_ap3_model():
    hirshfeld_model = AtomHirshfeldMPNN(
        # n_message=1,
        # n_rbf=4,
        # n_neuron=16,
        # n_embed=4,
    )
    atpnn_model = AtomTypeParamNN(
        atom_model=cast(AtomMPNN, hirshfeld_model),
        # n_message=1,
        # n_neuron=16,
        # n_embed=4,
        # n_params=1,
        freeze_atom_model=True,
    )
    dimer_prop_model = DimerProp(ATParam=atpnn_model, freeze_atom_model=True)
    return APNet3_AtomType_Model(
        dimer_prop_model=dimer_prop_model,
        use_GPU=False,
        ignore_database_null=True,
        # n_message=1,
        # n_rbf=4,
        # n_neuron=16,
        # n_embed=4,
    )


def build_demo_dapnet2_model():
    atom_model = AtomMPNN(
            # n_message=1, n_rbf=4, n_neuron=16, n_embed=4
            )
    apnet2_model = APNet2Model(
        atom_model=atom_model,
        use_GPU=False,
        ignore_database_null=True,
        # n_message=1,
        # n_rbf=4,
        # n_neuron=16,
        # n_embed=4,
    )
    return dAPNet2Model(
        apnet2_model=apnet2_model,
        atom_model=atom_model,
        use_GPU=False,
        ignore_database_null=True,
        # n_neuron=16,
    )


@pytest.fixture(scope="module")
def atom_mpnn():
    return AtomMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)


@pytest.fixture(scope="module")
def atom_model():
    return AtomModel(
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


@pytest.fixture(scope="module")
def hirshfeld():
    return AtomHirshfeldMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)


@pytest.fixture(scope="module")
def atpnn(hirshfeld):
    return AtomTypeParamNN(
        atom_model=hirshfeld,
        n_message=1,
        n_neuron=16,
        n_embed=4,
        n_params=1,
        freeze_atom_model=True,
    )


@pytest.fixture(scope="module")
def dimer_prop(atpnn):
    return DimerProp(ATParam=atpnn, freeze_atom_model=True)


@pytest.fixture(scope="module")
def apnet2_model(atom_mpnn):
    return APNet2Model(
        atom_model=atom_mpnn,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


@pytest.fixture(scope="module")
def ap3_model(dimer_prop):
    return APNet3_AtomType_Model(
        dimer_prop_model=dimer_prop,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


@pytest.fixture(scope="module")
def dapnet2_model(atom_mpnn, apnet2_model):
    return dAPNet2Model(
        apnet2_model=apnet2_model,
        atom_model=atom_mpnn,
        use_GPU=False,
        ignore_database_null=True,
        n_neuron=16,
    )


def test_atom_mpnn_model_info(atom_mpnn):
    info = atom_mpnn.get_model_info()
    assert info.name == "AtomMPNN"
    assert "q" in info.outputs
    assert "h_list" in info.outputs
    assert info.n_calls == 1
    assert info.n_params_total == count_params(atom_mpnn)


def test_atom_model_tree(atom_model):
    info = atom_model.get_model_info()
    assert info.name == "AtomModel"
    assert [child.name for child in info.children] == ["AtomMPNN"]
    assert info.n_params_total == count_params(atom_model.model)


def test_apnet2_mpnn_model_info():
    model = APNet2_MPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)
    info = model.get_model_info()
    assert info.name == "APNet2_MPNN"
    assert "E_exch" in ", ".join(info.outputs)
    assert info.call_note
    assert info.n_calls == 1


def test_apnet2_model_tree(apnet2_model):
    info = apnet2_model.get_model_info()
    child_names = [child.name for child in info.children]
    assert child_names == ["AtomMPNN", "APNet2_MPNN"]
    assert info.children[0].n_calls == 2
    assert info.n_params_total == count_params(apnet2_model.atom_model) + count_params(
        apnet2_model.model
    )


def test_ap3_model_tree(ap3_model):
    info = ap3_model.get_model_info()
    child_names = [child.name for child in info.children]
    assert "AtomHirshfeldMPNN" in child_names
    assert "AtomTypeParamNN" in child_names
    assert "Classical" in child_names
    assert "APNet3_AtomType_MPNN" in child_names
    classical = next(child for child in info.children if child.name == "Classical")
    assert classical.is_group is True
    assert {child.name for child in classical.children} == {
        "DampedMTPElectrostatics",
        "PointInducedDipole",
    }
    hirshfeld = next(child for child in info.children if child.name == "AtomHirshfeldMPNN")
    assert hirshfeld.n_calls == 2
    atpnn = next(child for child in info.children if child.name == "AtomTypeParamNN")
    assert atpnn.n_calls == 2
    assert atpnn.children == []
    assert info.n_params_total == count_params(ap3_model.model)


def test_dapnet2_model_tree(dapnet2_model):
    info = dapnet2_model.get_model_info()
    child_names = [child.name for child in info.children]
    assert child_names == ["AtomMPNN", "APNet2_MPNN", "dAPNet2_MPNN"]
    assert info.children[0].n_calls == 2
    assert info.n_params_total == (
        count_params(dapnet2_model.atom_model)
        + count_params(dapnet2_model.apnet2_model.model)
        + count_params(dapnet2_model.model)
    )


@pytest.mark.parametrize(
    ("target", "expected_name"),
    [
        ("atom_model", "AtomModel"),
        ("apnet2_model", "APNet2Model"),
        ("ap3_model", "APNet3_AtomType_Model"),
        ("dapnet2_model", "dAPNet2Model"),
    ],
)
def test_print_model_tree_smoke(request, target, expected_name):
    buf = io.StringIO()
    print_model_tree(request.getfixturevalue(target), file=buf, unicode=False)
    out = buf.getvalue()
    assert expected_name in out
    assert out.strip()


def test_get_model_info_top_level_dispatch(ap3_model):
    info = get_model_info(ap3_model)
    assert info.name == "APNet3_AtomType_Model"


if __name__ == "__main__":
    demo_models = [
        ("AtomModel", build_demo_atom_model()),
        ("APNet2Model", build_demo_apnet2_model()),
        ("APNet3_AtomType_Model", build_demo_ap3_model()),
        ("dAPNet2Model", build_demo_dapnet2_model()),
    ]

    for name, model in demo_models:
        print("\n" + "=" * 70)
        print(f"{name} Architecture Tree (ASCII)")
        print("=" * 70)
        print_model_tree(model, unicode=False)
        print("=" * 70)
