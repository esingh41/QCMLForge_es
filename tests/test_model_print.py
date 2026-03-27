"""
Tests for apnet_pt.model_print — ModelInfo, get_model_info, print_model_tree.

Covers the AP3D3 architecture tree:

  APNet3D3_AtomType_Model
  ├─ [frozen ×2] AtomHirshfeldMPNN
  ├─ [frozen ×2] AtomTypeParamNN
  ├─ Classical  (non-trainable)
  │   ├─ DampedMTPElectrostatics
  │   ├─ PointInducedDipole
  │   └─ DFTD3Dispersion
  └─ [train] APNet3D3_AtomType_MPNN
"""

import io
from typing import cast

import pytest
import apnet_pt
from apnet_pt import ModelInfo, get_model_info, print_model_tree
from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomModels.ap2_hirshfeld_atom_model import AtomHirshfeldMPNN
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
    APNet3D3_AtomType_Model,
    APNet3D3_AtomType_MPNN,
)
from apnet_pt.AtomPairwiseModels.mtp_mtp import AtomTypeParamNN, DimerProp
from torch.nn.parameter import UninitializedParameter


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hirshfeld():
    """Minimal AtomHirshfeldMPNN (untrained, CPU)."""
    return AtomHirshfeldMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)


@pytest.fixture(scope="module")
def atpnn(hirshfeld):
    """AtomTypeParamNN wrapping AtomHirshfeldMPNN with atom model frozen."""
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
    """DimerProp using the minimal AtomTypeParamNN."""
    return DimerProp(ATParam=atpnn, freeze_atom_model=True)


@pytest.fixture(scope="module")
def ap3d3_mpnn(dimer_prop):
    """APNet3D3_AtomType_MPNN (small, CPU, untrained)."""
    return APNet3D3_AtomType_MPNN(
        dimer_prop_model=dimer_prop,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


@pytest.fixture(scope="module")
def ap3d3_model(dimer_prop):
    """Full APNet3D3_AtomType_Model wrapper (no dataset, CPU)."""
    return APNet3D3_AtomType_Model(
        dimer_prop_model=dimer_prop,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


def build_demo_ap3d3_model():
    """Construct a minimal AP3D3 model for direct script execution."""
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
    return APNet3D3_AtomType_Model(
        dimer_prop_model=dimer_prop_model,
        use_GPU=False,
        ignore_database_null=True,
        # n_message=1,
        # n_rbf=4,
        # n_neuron=16,
        # n_embed=4,
    )


def count_params(module, trainable_only=False):
    """Count parameters, treating lazy uninitialized tensors as size zero."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(
        0 if isinstance(p, UninitializedParameter) else p.numel() for p in params
    )


# ── ModelInfo dataclass ────────────────────────────────────────────────────────


class TestModelInfo:
    def test_defaults(self):
        info = ModelInfo(name="Foo")
        assert info.name == "Foo"
        assert info.role == ""
        assert info.inputs == []
        assert info.outputs == []
        assert info.passes == []
        assert info.frozen is False
        assert info.n_params == 0
        assert info.n_params_total == 0
        assert info.n_calls == 1
        assert info.call_note == ""
        assert info.children == []
        assert info.is_group is False

    def test_mutable_defaults_are_independent(self):
        a = ModelInfo(name="A")
        b = ModelInfo(name="B")
        a.inputs.append("x")
        assert b.inputs == [], "Field default_factory must produce separate lists"

    def test_children_nesting(self):
        child = ModelInfo(name="Child", n_calls=2)
        parent = ModelInfo(name="Parent", children=[child])
        assert parent.children[0].n_calls == 2


# ── AtomHirshfeldMPNN.get_model_info ──────────────────────────────────────────


class TestAtomHirshfeldMPNNInfo:
    def test_n_calls_is_1(self, hirshfeld):
        info = hirshfeld.get_model_info()
        assert info.n_calls == 1, "AtomHirshfeldMPNN runs once per monomer batch"

    def test_name(self, hirshfeld):
        info = hirshfeld.get_model_info()
        assert info.name == "AtomHirshfeldMPNN"

    def test_inputs_contains_Z_and_R(self, hirshfeld):
        info = hirshfeld.get_model_info()
        assert "Z" in info.inputs
        assert "R" in info.inputs

    def test_outputs_contains_multipoles_and_embeddings(self, hirshfeld):
        info = hirshfeld.get_model_info()
        joined = ", ".join(info.outputs)
        assert "q" in joined
        assert "HFVR" in joined
        assert "VW" in joined
        assert "h_list" in joined

    def test_call_note_empty(self, hirshfeld):
        info = hirshfeld.get_model_info()
        assert info.call_note == ""

    def test_frozen_when_requires_grad_false(self, hirshfeld):
        # Freeze and check
        hirshfeld.requires_grad_(False)
        info = hirshfeld.get_model_info()
        assert info.frozen is True
        # Restore
        hirshfeld.requires_grad_(True)

    def test_param_counts_nonzero(self, hirshfeld):
        info = hirshfeld.get_model_info()
        assert info.n_params_total > 0

    def test_get_model_info_delegation(self, hirshfeld):
        """get_model_info() top-level function must delegate to the method."""
        info_method = hirshfeld.get_model_info()
        info_func = get_model_info(hirshfeld)
        assert info_method.name == info_func.name
        assert info_method.n_calls == info_func.n_calls


# ── AtomTypeParamNN.get_model_info ────────────────────────────────────────────


class TestAtomTypeParamNNInfo:
    def test_n_calls_is_1(self, atpnn):
        info = atpnn.get_model_info()
        assert info.n_calls == 1, "AtomTypeParamNN runs once per monomer batch"

    def test_name(self, atpnn):
        info = atpnn.get_model_info()
        assert info.name == "AtomTypeParamNN"

    def test_outputs_K(self, atpnn):
        info = atpnn.get_model_info()
        assert "K" in info.outputs

    def test_inputs_reference_runtime_atom_model_name(self, atpnn):
        info = atpnn.get_model_info()
        assert info.inputs == ["h_list [from AtomHirshfeldMPNN]"]

    def test_inputs_reference_atom_mpnn_when_used(self):
        atom_model = AtomMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)
        atpnn = AtomTypeParamNN(
            atom_model=atom_model,
            n_message=1,
            n_neuron=16,
            n_embed=4,
            n_params=1,
            freeze_atom_model=True,
        )
        info = atpnn.get_model_info()
        assert info.inputs == ["h_list [from AtomMPNN]"]

    def test_inputs_reference_nested_atomtypeparam_when_used(self):
        inner = AtomTypeParamNN(
            atom_model=AtomMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4),
            n_message=1,
            n_neuron=16,
            n_embed=4,
            n_params=1,
            freeze_atom_model=True,
        )
        outer = AtomTypeParamNN(
            atom_model=inner,
            n_message=1,
            n_neuron=16,
            n_embed=4,
            n_params=1,
            freeze_atom_model=True,
        )
        info = outer.get_model_info()
        assert info.inputs == ["h_list [from AtomTypeParamNN]"]

    def test_call_note_empty(self, atpnn):
        info = atpnn.get_model_info()
        assert info.call_note == ""

    def test_child_is_atom_hirshfeld(self, atpnn):
        info = atpnn.get_model_info()
        child_names = [c.name for c in info.children]
        assert "AtomHirshfeldMPNN" in child_names

    def test_child_hirshfeld_runs_once(self, atpnn):
        info = atpnn.get_model_info()
        child = next(c for c in info.children if c.name == "AtomHirshfeldMPNN")
        assert child.n_calls == 1

    def test_frozen_reflects_frozen_atom_model(self, atpnn):
        """AtomTypeParamNN with frozen atom_model should show frozen=True
        when all its parameters (including the frozen sub-model) are non-trainable."""
        info = atpnn.get_model_info()
        # atpnn was constructed with freeze_atom_model=True; the
        # param_readout_layers are trainable, so overall frozen=False
        # but atom_model child should be frozen
        child = next(c for c in info.children if c.name == "AtomHirshfeldMPNN")
        assert child.frozen is True


# ── APNet3D3_AtomType_MPNN.get_model_info ─────────────────────────────────────


class TestAPNet3D3AtomTypeMPNNInfo:
    def test_n_calls_is_1(self, ap3d3_mpnn):
        info = ap3d3_mpnn.get_model_info()
        assert info.n_calls == 1, "APNet3D3_AtomType_MPNN is called once per dimer"

    def test_call_note_explains_internal_dual_run(self, ap3d3_mpnn):
        info = ap3d3_mpnn.get_model_info()
        assert info.call_note, "call_note must explain internal dual monomer run"
        note_lower = info.call_note.lower()
        assert any(kw in note_lower for kw in ("intra", "monomer", "shared")), (
            f"call_note should mention intra-monomer shared-weight execution; got: {info.call_note!r}"
        )

    def test_name(self, ap3d3_mpnn):
        info = ap3d3_mpnn.get_model_info()
        assert info.name == "APNet3D3_AtomType_MPNN"

    def test_outputs_sapt_components(self, ap3d3_mpnn):
        info = ap3d3_mpnn.get_model_info()
        assert info.outputs[:3] == ["ΔE_elst_NN", "E_exch_NN", "ΔE_ind_NN"]
        assert info.outputs[3:] == ["ΔE_disp_NN"]

    def test_param_count_nonnegative(self, ap3d3_mpnn):
        """Param count may be 0 for an uninitalized LazyLinear model (no forward yet)."""
        info = ap3d3_mpnn.get_model_info()
        assert info.n_params_total >= 0
        assert info.n_params >= 0


# ── APNet3D3_AtomType_Model.get_model_info ────────────────────────────────────


class TestAPNet3D3AtomTypeModelInfo:
    def test_name(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        assert info.name == "APNet3D3_AtomType_Model"

    def test_has_four_top_level_children(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        child_names = [c.name for c in info.children]
        assert "AtomHirshfeldMPNN" in child_names
        assert "AtomTypeParamNN" in child_names
        assert "Classical" in child_names
        assert "APNet3D3_AtomType_MPNN" in child_names

    def test_hirshfeld_is_frozen_and_n_calls_2(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        hf = next(c for c in info.children if c.name == "AtomHirshfeldMPNN")
        assert hf.frozen is True, "AtomHirshfeldMPNN must be frozen in loaded AP3D3"
        assert hf.n_calls == 2

    def test_atpnn_is_n_calls_2(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        atpnn = next(c for c in info.children if c.name == "AtomTypeParamNN")
        assert atpnn.n_calls == 2

    def test_classical_is_group(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        cl = next(c for c in info.children if c.name == "Classical")
        assert cl.is_group is True

    def test_classical_has_three_sub_models(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        cl = next(c for c in info.children if c.name == "Classical")
        sub_names = {c.name for c in cl.children}
        assert "DampedMTPElectrostatics" in sub_names
        assert "PointInducedDipole" in sub_names
        assert "DFTD3Dispersion" in sub_names

    def test_mpnn_n_calls_is_1_with_call_note(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        mpnn = next(c for c in info.children if c.name == "APNet3D3_AtomType_MPNN")
        assert mpnn.n_calls == 1
        assert mpnn.call_note, "APNet3D3_AtomType_MPNN must have a call_note"

    def test_atpnn_flattened_no_nested_children(self, ap3d3_model):
        """AtomTypeParamNN in the assembled tree must not have nested children
        (the top-level view is flat — AtomHirshfeldMPNN is a sibling)."""
        info = ap3d3_model.get_model_info()
        atpnn = next(c for c in info.children if c.name == "AtomTypeParamNN")
        assert atpnn.children == [], (
            "In the APNet3D3_AtomType_Model tree, AtomTypeParamNN should be "
            "a sibling (not a parent) of AtomHirshfeldMPNN"
        )

    def test_root_total_matches_unique_model_parameters(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        assert info.n_params_total == count_params(ap3d3_model.model)

    def test_hirshfeld_total_matches_atom_model(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        hf = next(c for c in info.children if c.name == "AtomHirshfeldMPNN")
        atom_model = ap3d3_model.model.dimer_prop_model.AtomTypeParam.atom_model
        assert hf.n_params_total == count_params(atom_model)

    def test_atpnn_total_excludes_flattened_atom_model(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        atpnn = next(c for c in info.children if c.name == "AtomTypeParamNN")
        atom_model = ap3d3_model.model.dimer_prop_model.AtomTypeParam.atom_model
        at_param = ap3d3_model.model.dimer_prop_model.AtomTypeParam
        expected = count_params(at_param) - count_params(atom_model)
        assert atpnn.n_params_total == expected

    def test_mpnn_total_excludes_flattened_dimer_prop_model(self, ap3d3_model):
        info = ap3d3_model.get_model_info()
        mpnn = next(c for c in info.children if c.name == "APNet3D3_AtomType_MPNN")
        expected = count_params(ap3d3_model.model) - count_params(
            ap3d3_model.model.dimer_prop_model
        )
        assert mpnn.n_params_total == expected


# ── print_model_tree ──────────────────────────────────────────────────────────


class TestPrintModelTree:
    def _capture(self, target, **kwargs):
        buf = io.StringIO()
        print_model_tree(target, file=buf, **kwargs)
        return buf.getvalue()

    def test_unicode_output_nonempty(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert out.strip(), "print_model_tree must produce non-empty output"

    def test_root_name_in_output(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert "APNet3D3_AtomType_Model" in out

    def test_root_shows_total_param_count(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        first_line = out.splitlines()[0]
        expected = f"({count_params(ap3d3_model.model) / 1e3:.0f} k params)"
        assert "APNet3D3_AtomType_Model" in first_line
        assert expected in first_line

    def test_hirshfeld_shown_with_x2(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert "AtomHirshfeldMPNN" in out
        assert "×2" in out

    def test_atpnn_shown_with_x2(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert "AtomTypeParamNN" in out
        # Check that ×2 appears (may appear multiple times)
        assert out.count("×2") >= 2

    def test_classical_shown_as_non_trainable(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert "Classical" in out
        assert "non-trainable" in out

    def test_damped_electrostatics_in_output(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert "DampedMTPElectrostatics" in out

    def test_mpnn_call_note_in_output(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert "APNet3D3_AtomType_MPNN" in out
        assert "note:" in out

    def test_inputs_outputs_shown(self, ap3d3_model):
        out = self._capture(ap3d3_model)
        assert "Inputs" in out
        assert "Outputs" in out

    def test_unicode_box_chars(self, ap3d3_model):
        out = self._capture(ap3d3_model, unicode=True)
        assert "├─" in out or "└─" in out, (
            "Unicode mode must use box-drawing characters"
        )

    def test_ascii_fallback(self, ap3d3_model):
        out = self._capture(ap3d3_model, unicode=False)
        assert "├─" not in out, "ASCII mode must not use Unicode box characters"
        assert "+- " in out or "\\- " in out, "ASCII mode must use +- / \\- connectors"
        assert "APNet3D3_AtomType_Model" in out

    def test_ascii_output_nonempty(self, ap3d3_model):
        out = self._capture(ap3d3_model, unicode=False)
        assert out.strip()

    def test_accepts_modelinfo_directly(self):
        info = ModelInfo(
            name="Root",
            children=[
                ModelInfo(
                    name="Child",
                    role="test",
                    n_calls=2,
                    n_params=100,
                    n_params_total=100,
                    inputs=["x"],
                    outputs=["y"],
                ),
            ],
        )
        out = self._capture(info)
        assert "Root (100 params)" in out
        assert "Child" in out
        assert "×2" in out
        assert "x" in out
        assert "y" in out

    def test_accepts_nn_module_generic_fallback(self):
        """Generic nn.Module without get_model_info must not raise."""
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        out = self._capture(model)
        assert "Sequential" in out

    def test_generic_fallback_shows_children(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        info = get_model_info(model)
        child_names = [c.name for c in info.children]
        assert "Linear" in child_names
        assert "ReLU" in child_names

    def test_frozen_module_shows_frozen(self):
        import torch.nn as nn

        model = nn.Linear(4, 4)
        model.requires_grad_(False)
        info = get_model_info(model)
        assert info.frozen is True

    def test_print_to_custom_file(self, ap3d3_model):
        buf = io.StringIO()
        print_model_tree(ap3d3_model, file=buf)
        assert len(buf.getvalue()) > 0


# ── top-level imports ─────────────────────────────────────────────────────────


def test_imports_from_apnet_pt():
    """ModelInfo, get_model_info, print_model_tree must be importable from apnet_pt."""
    assert hasattr(apnet_pt, "ModelInfo")
    assert hasattr(apnet_pt, "get_model_info")
    assert hasattr(apnet_pt, "print_model_tree")


# ── visual demo ───────────────────────────────────────────────────────────────


def test_visual_tree_unicode(ap3d3_model, capsys):
    """Print the full tree to stdout so it appears in pytest -s output."""
    print("\n" + "=" * 70)
    print("AP3D3 Model Architecture Tree (Unicode)")
    print("=" * 70)
    print_model_tree(ap3d3_model)
    print("=" * 70)
    captured = capsys.readouterr()
    assert "APNet3D3_AtomType_Model" in captured.out


def test_visual_tree_ascii(ap3d3_model, capsys):
    """Print the ASCII fallback tree."""
    print("\n" + "=" * 70)
    print("AP3D3 Model Architecture Tree (ASCII)")
    print("=" * 70)
    print_model_tree(ap3d3_model, unicode=False)
    print("=" * 70)
    captured = capsys.readouterr()
    assert "APNet3D3_AtomType_Model" in captured.out


if __name__ == "__main__":
    demo_model = build_demo_ap3d3_model()
    print("\n" + "=" * 70)
    print("AP3D3 Model Architecture Tree (ASCII)")
    print("=" * 70)
    print_model_tree(demo_model, unicode=False)
    print("=" * 70)

    print("\n" + "=" * 70)
    print("AP3D3 Model Architecture Tree (Unicode)")
    print("=" * 70)
    print_model_tree(demo_model)
    print("=" * 70)

    # pytest.main([__file__, "-s"])
