from train_saptdft import (
    build_local_dry_run_artifacts,
    build_local_dry_run_commands,
    build_bash_training_body,
    build_training_body,
)


def _wrapped(flag: str, value: str) -> str:
    return flag + " \\" + "\n    " + value


def test_build_training_body_uses_staged_ap2_atomtype_ap3_flow():
    body = build_training_body(
        iteration=3,
        atom_data_dir="./atom_data",
        splinter_data_dir_template="./splinter_{iter}",
        saptpbe0_data_dir="./saptpbe0",
        model_root="./models/ap3_saptpbe0",
    )

    assert "# AP2 AtomMPNN on PBE0 monomers" in body
    assert "# Hirshfeld volume-ratio/valence-width AtomTypeParamNN" in body
    assert "# Electrostatic K AtomTypeParamNN on Splinter" in body
    assert (
        "# APNet3D3 on Splinter SAPT0/aug-cc-pVDZ (spec 2), with -D3 + NN "
        "dispersion"
    ) in body

    assert "./models/ap3_saptpbe0/3/am_ap2_3.pt" in body
    assert "./models/ap3_saptpbe0/3/atp_hfvr_3.pt" in body
    assert "./models/ap3_saptpbe0/3/atp_elst_3.pt" in body
    assert "./models/ap3_saptpbe0/3/ap3d3_3.pt" in body

    assert _wrapped("--train_am", "AtomModel") in body
    assert _wrapped("--spec_type_am", "4") in body

    assert body.count(_wrapped("--train_apnet", "AtomTypeParamModel")) >= 2
    assert _wrapped("--spec_type_ap", "1") in body
    assert _wrapped("--train_apnet", "AM-DimerParam") in body
    assert body.count(_wrapped("--spec_type_ap", "2")) >= 3
    assert _wrapped("--train_apnet", "APNet3-fused-d3") in body

    assert body.index("am_ap2_3.pt") < body.index("atp_hfvr_3.pt")
    assert body.index("atp_hfvr_3.pt") < body.index("atp_elst_3.pt")
    assert body.index("atp_elst_3.pt") < body.index("ap3d3_3.pt")


def test_build_training_body_syncs_expected_raw_files():
    body = build_training_body(
        iteration=1,
        atom_data_dir="./atom_data",
        splinter_data_dir_template="./splinter_{iter}",
        saptpbe0_data_dir="./saptpbe0",
        model_root="./models/ap3_saptpbe0",
    )

    assert body.count("monomers_ap3_spec_1_pbe0.pkl") >= 2
    assert "1600K_train_dimers-fixed.pkl" in body
    assert "1600K_test_dimers-fixed.pkl" in body
    assert "124K_saptpbe0-d4_totals_train.pkl" in body
    assert "124K_saptpbe0-d4_totals_test.pkl" in body
    assert (
        "cp ./models/ap3_saptpbe0/1/ap3d3_1.pt "
        "./models/ap3_saptpbe0/1/ap3d3_1_saptpbe0.pt"
    ) in body


def test_local_dry_run_plan_uses_small_dataset_equivalents(tmp_path):
    test_data_dir = tmp_path / "test_data"
    raw_dir = test_data_dir / "raw"
    raw_dir.mkdir(parents=True)
    for name in [
        "monomers_ap3_spec_5_pbe0.pkl",
        "t_train_100.pkl",
        "t_test_20.pkl",
    ]:
        (raw_dir / name).write_text("placeholder")

    artifacts = build_local_dry_run_artifacts(
        work_root=tmp_path / "work",
        test_data_dir=test_data_dir,
        iteration=2,
    )
    commands = build_local_dry_run_commands(artifacts, random_seed=11)

    assert len(commands) == 4
    assert artifacts.atom_data_dir.joinpath("raw", "monomers_ap3_spec_5_pbe0.pkl").is_file()
    assert artifacts.dimer_data_dir.joinpath("raw", "t_train_100.pkl").is_file()
    assert artifacts.dimer_data_dir.joinpath("raw", "t_test_20.pkl").is_file()

    command_text = "\n".join(command for _, command in commands)
    assert _wrapped("--spec_type_am", "9") in command_text
    assert _wrapped("--spec_type_ap", "5") in command_text
    assert command_text.count(_wrapped("--spec_type_ap", "7")) >= 2
    assert str(artifacts.atom_model_path) in command_text
    assert str(artifacts.hirshfeld_model_path) in command_text
    assert str(artifacts.elst_model_path) in command_text
    assert str(artifacts.ap3d3_model_path) in command_text


def test_build_bash_training_body_skips_dataset_only_setup():
    body = build_bash_training_body(
        iteration=4,
        atom_data_dir="./atom_data",
        splinter_data_dir_template="./splinter_{iter}",
        saptpbe0_data_dir="./saptpbe0",
        model_root="./models/ap3_saptpbe0",
    )

    assert _wrapped("--train_am", "AtomModel") in body
    assert _wrapped("--train_apnet", "AtomTypeParamModel") in body
    assert _wrapped("--train_apnet", "AM-DimerParam") in body
    assert _wrapped("--train_apnet", "APNet3-fused-d3") in body

    assert "--build_dataset_only" not in body
    assert "sync_dataset_to_tmp" not in body
    assert "SCRATCH_ROOT" not in body
    assert 'mkdir -p "${MODEL_DIR}"' in body
    assert "./splinter_4" in body
