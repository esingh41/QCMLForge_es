"""
Tests for freeze/unfreeze behavior of APNet3 submodels.

Tests the freeze chain:
  - DimerProp(freeze_atom_model=True/False)
  - AtomTypeParamNN(freeze_atom_model=True/False)
  - AM_DimerParam_Model(freeze_atom_model=True/False) -> DimerProp
  - AtomTypeParamModel(freeze_atom_model=True/False) -> AtomTypeParamNN
  - APNet3_AtomType_MPNN(freeze_dimer_prop_model=True/False)
  - APNet3D3_AtomType_MPNN(freeze_dimer_prop_model=True/False)
  - Config persistence (get_config round-trip)
"""

import os
from typing import Any, cast

import qcelemental as qcel
import torch
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
    APNet3D3_AtomType_MPNN,
)
from apnet_pt.AtomPairwiseModels.apnet3_fused import (
    APNet3_AtomType_MPNN,
)
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    AM_DimerParam_Model,
    AtomTypeParamModel,
    AtomTypeParamNN,
    DimerProp,
)
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_collate_update_no_target,
    qcel_dimer_to_fused_data,
)

torch.manual_seed(42)

current_file_path = os.path.dirname(os.path.realpath(__file__))
am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_atom_type_hf_vw_model(**kwargs: Any) -> Any:
    """Build an AtomTypeParamModel with default test paths."""
    defaults: dict[str, Any] = dict(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    defaults.update(kwargs)
    return AtomTypeParamModel(**defaults)


def _build_atom_type_elst_model(atom_type_hf_vw_model: Any, **kwargs: Any) -> Any:
    """Build an AM_DimerParam_Model with default test paths."""
    defaults: dict[str, Any] = dict(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    defaults.update(kwargs)
    return AM_DimerParam_Model(**defaults)


def _all_params_frozen(module):
    """Return True if every parameter in module has requires_grad=False."""
    params = list(module.parameters())
    if len(params) == 0:
        return True
    return all(not p.requires_grad for p in params)


def _all_params_trainable(module):
    """Return True if every parameter in module has requires_grad=True."""
    params = list(module.parameters())
    if len(params) == 0:
        return True
    return all(p.requires_grad for p in params)


def _has_any_trainable(module):
    """Return True if at least one parameter in module has requires_grad=True."""
    return any(p.requires_grad for p in module.parameters())


def _has_any_frozen(module):
    """Return True if at least one parameter in module has requires_grad=False."""
    return any(not p.requires_grad for p in module.parameters())


def _make_ap3d3_batch():
    mol = qcel.models.Molecule.from_data(
        """
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224    0.846775782  -0.011488714
1    0.257521062    0.042121496   0.005218999
--
0 1
8    2.268880784    0.026340101   0.000508029
1    2.645502399   -0.412039965   0.766632411
1    2.641145101   -0.449872874  -0.744894473
units angstrom
"""
    )
    batch = ap3_fused_collate_update_no_target(
        [qcel_dimer_to_fused_data(mol, dimer_ind=0)]
    )
    batch.y = torch.zeros((1, 4), dtype=torch.float32)
    return batch


def _randomize_initialized_parameters(module, seed=42, scale=0.05):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                continue
            random_values = torch.randn(
                param.shape,
                generator=generator,
                device=param.device,
                dtype=param.dtype,
            )
            param.copy_(random_values * scale)


def _clone_named_params(module, predicate=None):
    snapshots = {}
    for name, param in module.named_parameters():
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            continue
        if predicate is not None and not predicate(name, param):
            continue
        snapshots[name] = param.detach().clone()
    return snapshots


def _any_param_changed(module, before, predicate=None):
    for name, param in module.named_parameters():
        if name not in before:
            continue
        if predicate is not None and not predicate(name, param):
            continue
        if not torch.equal(param.detach(), before[name]):
            return True
    return False


# ===========================================================================
# 1. DimerProp freeze_atom_model
# ===========================================================================


class TestDimerPropFreezeAtomModel:
    """Test DimerProp(freeze_atom_model=...) behavior."""

    def test_freeze_atom_model_default(self):
        """Default freeze_atom_model=True should freeze atom_model inside AtomTypeParam."""
        hf_vw = _build_atom_type_hf_vw_model()
        dimer = DimerProp(ATParam=hf_vw.model, freeze_atom_model=True)
        assert _all_params_frozen(dimer.AtomTypeParam.atom_model), (
            "atom_model parameters should be frozen when freeze_atom_model=True"
        )

    def test_unfreeze_atom_model(self):
        """freeze_atom_model=False should leave atom_model trainable."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        dimer = DimerProp(ATParam=hf_vw.model, freeze_atom_model=False)
        assert _all_params_trainable(dimer.AtomTypeParam.atom_model), (
            "atom_model parameters should be trainable when freeze_atom_model=False"
        )


# ===========================================================================
# 2. AtomTypeParamNN freeze_atom_model
# ===========================================================================


class TestAtomTypeParamNNFreezeAtomModel:
    """Test AtomTypeParamNN(freeze_atom_model=...) behavior."""

    def test_freeze_atom_model_default(self):
        """Default freeze_atom_model=True should freeze self.atom_model."""
        hf_vw = _build_atom_type_hf_vw_model()
        nn_model = AtomTypeParamNN(
            atom_model=hf_vw.model.atom_model,
            freeze_atom_model=True,
        )
        assert _all_params_frozen(nn_model.atom_model), (
            "atom_model should be frozen when freeze_atom_model=True"
        )

    def test_unfreeze_atom_model(self):
        """freeze_atom_model=False should leave atom_model trainable."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        nn_model = AtomTypeParamNN(
            atom_model=hf_vw.model.atom_model,
            freeze_atom_model=False,
        )
        assert _all_params_trainable(nn_model.atom_model), (
            "atom_model should be trainable when freeze_atom_model=False"
        )


# ===========================================================================
# 3. AM_DimerParam_Model passes freeze_atom_model through to DimerProp
# ===========================================================================


class TestAMDimerParamModelFreezePassthrough:
    """Test AM_DimerParam_Model(freeze_atom_model=...) passes through to DimerProp."""

    def test_freeze_passthrough_frozen(self):
        """freeze_atom_model=True should propagate to dimer_model's atom_model."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=True)
        assert _all_params_frozen(elst.dimer_model.AtomTypeParam.atom_model), (
            "dimer_model.AtomTypeParam.atom_model should be frozen"
        )

    def test_freeze_passthrough_unfrozen(self):
        """
        freeze_atom_model=False should keep the full AtomTypeParam stack trainable.
        """
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=False)
        assert _all_params_trainable(elst.dimer_model.AtomTypeParam), (
            "elst AtomTypeParam should be trainable throughout when "
            "freeze_atom_model=False"
        )
        assert _all_params_trainable(elst.dimer_model.AtomTypeParam.atom_model), (
            "nested atom_model should also stay trainable when freeze_atom_model=False"
        )


# ===========================================================================
# 4. AtomTypeParamModel passes freeze_atom_model through to AtomTypeParamNN
# ===========================================================================


class TestAtomTypeParamModelFreezePassthrough:
    """Test AtomTypeParamModel(freeze_atom_model=...) passes through to AtomTypeParamNN."""

    def test_freeze_passthrough_frozen(self):
        """freeze_atom_model=True should propagate to self.model.atom_model."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=True)
        assert _all_params_frozen(hf_vw.model.atom_model), (
            "model.atom_model should be frozen"
        )

    def test_freeze_passthrough_unfrozen(self):
        """freeze_atom_model=False should propagate to self.model.atom_model."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        assert _all_params_trainable(hf_vw.model.atom_model), (
            "model.atom_model should be trainable"
        )


# ===========================================================================
# 5. APNet3_AtomType_MPNN freeze_dimer_prop_model
# ===========================================================================


class TestAPNet3FusedMPNNFreezeDimerPropModel:
    """Test APNet3_AtomType_MPNN(freeze_dimer_prop_model=...) behavior."""

    def test_freeze_dimer_prop_model_default(self):
        """Default freeze_dimer_prop_model=True should freeze all dimer_prop_model params."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        assert _all_params_frozen(model.dimer_prop_model), (
            "All dimer_prop_model params should be frozen when freeze_dimer_prop_model=True"
        )

    def test_unfreeze_dimer_prop_model(self):
        """freeze_dimer_prop_model=False should leave dimer_prop_model trainable."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )
        assert _has_any_trainable(model.dimer_prop_model), (
            "dimer_prop_model should have trainable params when freeze_dimer_prop_model=False"
        )

    def test_freeze_with_none_dimer_prop_model(self):
        """freeze_dimer_prop_model=True with dimer_prop_model=None should not raise."""
        model = APNet3_AtomType_MPNN(
            dimer_prop_model=cast(Any, None),
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        assert model.dimer_prop_model is None
        assert model.freeze_dimer_prop_model is True

    def test_config_contains_freeze_flag(self):
        """get_config() should contain freeze_dimer_prop_model."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        for flag_value in [True, False]:
            model = APNet3_AtomType_MPNN(
                dimer_prop_model=elst.dimer_model,
                use_precomputed_classical=True,
                freeze_dimer_prop_model=flag_value,
            )
            config = model.get_config()
            assert "freeze_dimer_prop_model" in config
            assert config["freeze_dimer_prop_model"] is flag_value

    def test_config_roundtrip(self):
        """Model recreated from get_config() should preserve freeze_dimer_prop_model."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )
        config = model.get_config()
        model2 = APNet3_AtomType_MPNN(
            dimer_prop_model=cast(Any, None),
            **config,
        )
        config2 = model2.get_config()
        assert config == config2


# ===========================================================================
# 6. APNet3D3_AtomType_MPNN freeze_dimer_prop_model
# ===========================================================================


class TestAPNet3D3FusedMPNNFreezeDimerPropModel:
    """Test APNet3D3_AtomType_MPNN(freeze_dimer_prop_model=...) behavior."""

    def test_freeze_dimer_prop_model_default(self):
        """Default freeze_dimer_prop_model=True should freeze all dimer_prop_model params."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        assert _all_params_frozen(model.dimer_prop_model), (
            "All dimer_prop_model params should be frozen when freeze_dimer_prop_model=True"
        )

    def test_unfreeze_dimer_prop_model(self):
        """freeze_dimer_prop_model=False should leave dimer_prop_model trainable."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )
        assert _has_any_trainable(model.dimer_prop_model), (
            "dimer_prop_model should have trainable params when freeze_dimer_prop_model=False"
        )

    def test_freeze_with_none_dimer_prop_model(self):
        """freeze_dimer_prop_model=True with dimer_prop_model=None should not raise."""
        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=cast(Any, None),
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        assert model.dimer_prop_model is None
        assert model.freeze_dimer_prop_model is True

    def test_config_contains_freeze_flag(self):
        """get_config() should contain freeze_dimer_prop_model."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        for flag_value in [True, False]:
            model = APNet3D3_AtomType_MPNN(
                dimer_prop_model=elst.dimer_model,
                use_precomputed_classical=True,
                freeze_dimer_prop_model=flag_value,
            )
            config = model.get_config()
            assert "freeze_dimer_prop_model" in config
            assert config["freeze_dimer_prop_model"] is flag_value

    def test_config_roundtrip(self):
        """Model recreated from get_config() should preserve freeze_dimer_prop_model."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)
        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
            no_disp_nn=True,
        )
        config = model.get_config()
        model2 = APNet3D3_AtomType_MPNN(
            dimer_prop_model=cast(Any, None),
            **config,
        )
        config2 = model2.get_config()
        assert config == config2


# ===========================================================================
# 7. Combined freeze behavior: freeze_dimer_prop_model + freeze_atom_model
# ===========================================================================


class TestCombinedFreezeBehavior:
    """Test the full freeze chain from harness through to atom_model parameters."""

    def test_both_frozen_by_default(self):
        """Default: both dimer_prop_model and atom_model should be frozen."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=True)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=True)

        # APNet3-fused
        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        assert _all_params_frozen(model.dimer_prop_model), (
            "dimer_prop_model should be entirely frozen"
        )

    def test_dimer_prop_frozen_atom_model_unfrozen(self):
        """
        freeze_dimer_prop_model=True but freeze_atom_model=False.

        The dimer_prop_model freeze in the MPNN sets requires_grad=False on all
        dimer_prop_model params, which overrides the atom_model's unfrozen state.
        So after MPNN construction, all dimer_prop_model params are frozen.
        """
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=False)

        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        # The MPNN freeze overrides atom_model's unfrozen state
        assert _all_params_frozen(model.dimer_prop_model), (
            "dimer_prop_model freeze should override atom_model's unfrozen state"
        )

    def test_dimer_prop_unfrozen_atom_model_frozen(self):
        """
        freeze_dimer_prop_model=False but freeze_atom_model=True.

        The atom_model was frozen during DimerProp construction.
        The MPNN does NOT re-freeze dimer_prop_model, so the atom_model stays frozen
        while other dimer_prop_model params remain trainable.
        """
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=True)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=True)

        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )
        # atom_model should still be frozen
        assert _all_params_frozen(model.dimer_prop_model.AtomTypeParam.atom_model), (
            "atom_model should remain frozen even when dimer_prop_model is unfrozen"
        )
        # But the overall dimer_prop_model should have SOME trainable params
        # (the non-atom-model params like param embeddings)
        assert _has_any_trainable(model.dimer_prop_model), (
            "dimer_prop_model should have trainable params when freeze_dimer_prop_model=False"
        )

    def test_both_unfrozen(self):
        """
        Both dimer_prop_model and atom_model unfrozen.

        freeze_dimer_prop_model=False means MPNN doesn't freeze dimer_prop_model.
        freeze_atom_model=False means the nested AtomTypeParamNN stack also stays
        trainable end-to-end.
        """
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=False)

        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )
        assert _all_params_trainable(model.dimer_prop_model), (
            "dimer_prop_model should stay trainable throughout when both freezes are False"
        )


# ===========================================================================
# 8. Same combined tests for D3 variant
# ===========================================================================


class TestCombinedFreezeBehaviorD3:
    """Test the full freeze chain for APNet3D3_AtomType_MPNN."""

    def test_both_frozen_by_default(self):
        """Default: both dimer_prop_model and atom_model should be frozen."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=True)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=True)

        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        assert _all_params_frozen(model.dimer_prop_model), (
            "dimer_prop_model should be entirely frozen"
        )

    def test_both_unfrozen(self):
        """
        Both dimer_prop_model and atom_model unfrozen (D3 variant).

        Same behavior as non-D3: the nested AtomTypeParamNN stack remains
        trainable throughout.
        """
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=False)

        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )
        assert _all_params_trainable(model.dimer_prop_model), (
            "dimer_prop_model should stay trainable throughout when both freezes are False"
        )

    def test_dimer_prop_frozen_overrides_atom_unfrozen(self):
        """freeze_dimer_prop_model=True overrides freeze_atom_model=False."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=False)

        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )
        assert _all_params_frozen(model.dimer_prop_model), (
            "dimer_prop_model freeze should override atom_model's unfrozen state"
        )


# ===========================================================================
# 9. Gradient flow verification
# ===========================================================================


class TestGradientFlow:
    """Verify that frozen parameters do not accumulate gradients."""

    def test_frozen_dimer_prop_no_grad(self):
        """Frozen dimer_prop_model params should not accumulate gradients after backward."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)

        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )

        # Verify no param in dimer_prop_model has requires_grad
        for name, param in model.dimer_prop_model.named_parameters():
            assert not param.requires_grad, (
                f"Frozen param {name} should have requires_grad=False"
            )

    def test_unfrozen_dimer_prop_has_grad(self):
        """Unfrozen dimer_prop_model params should have requires_grad=True."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=False)

        model = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )

        # At least some params should require grad
        trainable_params = [
            name
            for name, p in model.dimer_prop_model.named_parameters()
            if p.requires_grad
        ]
        assert len(trainable_params) > 0, (
            "Unfrozen dimer_prop_model should have at least one trainable parameter"
        )

    def test_frozen_atom_model_no_grad(self):
        """Frozen atom_model inside DimerProp should not have requires_grad."""
        hf_vw = _build_atom_type_hf_vw_model()
        dimer = DimerProp(ATParam=hf_vw.model, freeze_atom_model=True)

        for name, param in dimer.AtomTypeParam.atom_model.named_parameters():
            assert not param.requires_grad, f"Frozen atom_model param {
                name
            } should have requires_grad=False"

    def test_unfrozen_atom_model_has_grad(self):
        """Unfrozen atom_model inside DimerProp should have requires_grad."""
        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        dimer = DimerProp(ATParam=hf_vw.model, freeze_atom_model=False)

        trainable_params = [
            name
            for name, p in dimer.AtomTypeParam.atom_model.named_parameters()
            if p.requires_grad
        ]
        assert len(trainable_params) > 0, (
            "Unfrozen atom_model should have at least one trainable parameter"
        )


# ===========================================================================
# 10. Optimizer parameter count verification
# ===========================================================================


class TestOptimizerParamCounts:
    """Verify that freeze flags affect the number of trainable parameters seen by an optimizer."""

    def test_frozen_reduces_trainable_count(self):
        """Freezing dimer_prop_model should reduce the number of trainable params in MPNN."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)

        model_frozen = APNet3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )

        hf_vw2 = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst2 = _build_atom_type_elst_model(hf_vw2, freeze_atom_model=False)

        model_unfrozen = APNet3_AtomType_MPNN(
            dimer_prop_model=elst2.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )

        n_trainable_frozen = sum(
            p.numel()
            for p in model_frozen.parameters()
            if p.requires_grad
            and not isinstance(p, torch.nn.parameter.UninitializedParameter)
        )
        n_trainable_unfrozen = sum(
            p.numel()
            for p in model_unfrozen.parameters()
            if p.requires_grad
            and not isinstance(p, torch.nn.parameter.UninitializedParameter)
        )

        assert n_trainable_unfrozen > n_trainable_frozen, (
            f"Unfrozen model should have more trainable params "
            f"({n_trainable_unfrozen}) than frozen model ({n_trainable_frozen})"
        )

    def test_frozen_reduces_trainable_count_d3(self):
        """Freezing dimer_prop_model should reduce trainable params in D3 MPNN too."""
        hf_vw = _build_atom_type_hf_vw_model()
        elst = _build_atom_type_elst_model(hf_vw)

        model_frozen = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=True,
        )

        hf_vw2 = _build_atom_type_hf_vw_model(freeze_atom_model=False)
        elst2 = _build_atom_type_elst_model(hf_vw2, freeze_atom_model=False)

        model_unfrozen = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst2.dimer_model,
            use_precomputed_classical=True,
            freeze_dimer_prop_model=False,
        )

        n_trainable_frozen = sum(
            p.numel()
            for p in model_frozen.parameters()
            if p.requires_grad
            and not isinstance(p, torch.nn.parameter.UninitializedParameter)
        )
        n_trainable_unfrozen = sum(
            p.numel()
            for p in model_unfrozen.parameters()
            if p.requires_grad
            and not isinstance(p, torch.nn.parameter.UninitializedParameter)
        )

        assert n_trainable_unfrozen > n_trainable_frozen, (
            f"Unfrozen D3 model should have more trainable params "
            f"({n_trainable_unfrozen}) than frozen D3 model ({n_trainable_frozen})"
        )


# ===========================================================================
# 11. Train-step verification for AP3D3 fine-tuning freeze behavior
# ===========================================================================


class TestAP3D3TrainStepFreezeBehavior:
    def test_train_step_keeps_atomhirshfeld_frozen_but_updates_atomtype_and_ap3d3(self):
        """
        Mimic AP3D3 fine-tuning:

        - AtomHirshfeldMPNN stays frozen
        - the DampedMTPElectrostatics AtomTypeParamNN stays trainable
        - AP3D3 short-range MPNN stays trainable
        """
        torch.manual_seed(7)

        hf_vw = _build_atom_type_hf_vw_model(freeze_atom_model=True)
        elst = _build_atom_type_elst_model(hf_vw, freeze_atom_model=False)
        elst.dimer_model.set_forward("ap3_elst_damping__induced_dipole__disp")

        model = APNet3D3_AtomType_MPNN(
            dimer_prop_model=elst.dimer_model,
            n_message=1,
            n_rbf=4,
            n_neuron=16,
            n_embed=4,
            use_precomputed_classical=False,
            freeze_dimer_prop_model=False,
            no_disp_nn=False,
        )

        batch = _make_ap3d3_batch()

        with torch.no_grad():
            model(batch)
        _randomize_initialized_parameters(model, seed=11)

        atom_type_param = model.dimer_prop_model.AtomTypeParam
        atom_hirshfeld = atom_type_param.atom_model.atom_model

        hirshfeld_before = _clone_named_params(atom_hirshfeld)
        atomtype_before = _clone_named_params(
            atom_type_param,
            predicate=lambda name, param: param.requires_grad
            and not name.startswith("atom_model."),
        )
        ap3d3_before = _clone_named_params(
            model,
            predicate=lambda name, param: param.requires_grad
            and not name.startswith("dimer_prop_model."),
        )

        assert hirshfeld_before, "Expected AtomHirshfeldMPNN to have initialized params"
        assert atomtype_before, "Expected AtomTypeParamNN to expose trainable params"
        assert ap3d3_before, "Expected AP3D3 readout/message-passing params"

        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=1e-2,
        )
        target = cast(torch.Tensor, batch.y)

        for _ in range(2):
            optimizer.zero_grad()
            preds, *_ = model(batch)
            loss = torch.nn.functional.mse_loss(preds, target)
            loss.backward()
            optimizer.step()

        assert all(not param.requires_grad for param in atom_hirshfeld.parameters()), (
            "AtomHirshfeldMPNN should remain frozen"
        )
        assert all(param.grad is None for param in atom_hirshfeld.parameters()), (
            "Frozen AtomHirshfeldMPNN params should not accumulate gradients"
        )
        assert _any_param_changed(atom_hirshfeld, hirshfeld_before) is False, (
            "Frozen AtomHirshfeldMPNN params should not change during training"
        )

        assert _any_param_changed(
            atom_type_param,
            atomtype_before,
            predicate=lambda name, param: param.requires_grad
            and not name.startswith("atom_model."),
        ), "AtomTypeParamNN trainable params should update during training"

        assert _any_param_changed(
            model,
            ap3d3_before,
            predicate=lambda name, param: param.requires_grad
            and not name.startswith("dimer_prop_model."),
        ), "AP3D3 trainable params should update during training"
