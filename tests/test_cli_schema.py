"""Schema-validation tests for the qibochem CLI input schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qibochem.cli.schema import (
    ConfigError,
    HFMethodConfig,
    TopLevelConfig,
    VQEMethodConfig,
)


def _hf_dict() -> dict:
    return {
        "version": 1,
        "molecule": {
            "geometry": [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.74804]]],
            "basis": "sto-3g",
        },
        "hamiltonian": {"mapping": "jw"},
        "method": {"kind": "hf"},
        "output": {"log_level": "info"},
    }


def _vqe_dict() -> dict:
    d = _hf_dict()
    d["method"] = {
        "kind": "vqe",
        "ansatz": {"kind": "ucc", "excitations": ["singles", "doubles"]},
        "optimizer": {"name": "BFGS", "options": {"maxiter": 50}, "initial_parameters": "zeros", "seed": 0},
        "n_shots": None,
    }
    return d


def test_valid_hf_minimal():
    cfg = TopLevelConfig.parse(_hf_dict())
    assert isinstance(cfg.method, HFMethodConfig)
    assert cfg.molecule.basis == "sto-3g"
    assert cfg.hamiltonian.mapping == "jw"


def test_valid_vqe_full():
    cfg = TopLevelConfig.parse(_vqe_dict())
    assert isinstance(cfg.method, VQEMethodConfig)
    assert cfg.method.ansatz.kind == "ucc"
    assert cfg.method.optimizer.name == "BFGS"
    assert cfg.method.optimizer.options["maxiter"] == 50


def test_xyz_xor_geometry_required():
    d = _hf_dict()
    d["molecule"]["xyz_file"] = "h2.xyz"  # plus geometry already set
    with pytest.raises(ConfigError, match="exactly one of"):
        TopLevelConfig.parse(d)
    d["molecule"].pop("geometry")
    d["molecule"].pop("xyz_file")
    with pytest.raises(ConfigError, match="exactly one of"):
        TopLevelConfig.parse(d)


def test_invalid_mapping_rejected():
    d = _hf_dict()
    d["hamiltonian"]["mapping"] = "parity"
    with pytest.raises(ConfigError, match="hamiltonian.mapping"):
        TopLevelConfig.parse(d)


def test_unknown_method_kind_rejected():
    d = _hf_dict()
    d["method"] = {"kind": "monte-carlo"}
    with pytest.raises(ConfigError, match="method.kind"):
        TopLevelConfig.parse(d)


def test_qse_method_phase3_message():
    d = _hf_dict()
    d["method"] = {"kind": "qse"}
    with pytest.raises(ConfigError, match="not available in this build"):
        TopLevelConfig.parse(d)


def test_qsci_method_phase3_message():
    d = _hf_dict()
    d["method"] = {"kind": "qsci"}
    with pytest.raises(ConfigError, match="not available in this build"):
        TopLevelConfig.parse(d)


def test_invalid_ansatz_kind_rejected():
    d = _vqe_dict()
    d["method"]["ansatz"]["kind"] = "transformer"
    with pytest.raises(ConfigError, match="method.ansatz.kind"):
        TopLevelConfig.parse(d)


def test_invalid_excitation_label_rejected():
    d = _vqe_dict()
    d["method"]["ansatz"]["excitations"] = ["singles", "triples"]
    with pytest.raises(ConfigError, match="invalid entries"):
        TopLevelConfig.parse(d)


def test_multiplicity_must_be_positive():
    d = _hf_dict()
    d["molecule"]["multiplicity"] = 0
    with pytest.raises(ConfigError, match="multiplicity"):
        TopLevelConfig.parse(d)


def test_invalid_log_level_rejected():
    d = _hf_dict()
    d["output"]["log_level"] = "trace"
    with pytest.raises(ConfigError, match="output.log_level"):
        TopLevelConfig.parse(d)


def test_unknown_top_level_key_rejected():
    d = _hf_dict()
    d["mystery_field"] = 42
    with pytest.raises(ConfigError, match="unknown keys"):
        TopLevelConfig.parse(d)


def test_unsupported_version_rejected():
    d = _hf_dict()
    d["version"] = 2
    with pytest.raises(ConfigError, match="schema version"):
        TopLevelConfig.parse(d)


def test_initial_parameters_explicit_list_validated():
    d = _vqe_dict()
    d["method"]["optimizer"]["initial_parameters"] = [0.1, 0.2, "oops"]
    with pytest.raises(ConfigError, match="must contain only numbers"):
        TopLevelConfig.parse(d)


def test_initial_parameters_string_strategy_validated():
    d = _vqe_dict()
    d["method"]["optimizer"]["initial_parameters"] = "ones"
    with pytest.raises(ConfigError, match="initial_parameters"):
        TopLevelConfig.parse(d)


def test_n_shots_must_be_positive():
    d = _vqe_dict()
    d["method"]["n_shots"] = 0
    with pytest.raises(ConfigError, match="n_shots"):
        TopLevelConfig.parse(d)


def test_template_yaml_files_parse(tmp_path: Path):
    """The bundled templates must validate against the schema they document."""
    from importlib import resources

    for kind in ("hf", "vqe"):
        text = resources.files("qibochem.cli.templates").joinpath(f"{kind}.yaml").read_text(encoding="utf-8")
        d = yaml.safe_load(text)
        TopLevelConfig.parse(d)  # raises if invalid


def test_pes_scan_template_renders_to_valid_yaml():
    """The PES-scan driver embeds a YAML template with str.format escapes (`{{}}`).
    Lock it against schema drift: a rendered YAML must parse against the schema."""
    import importlib.util

    runner = Path(__file__).resolve().parent.parent / "examples/cli/pes_scan/run_scan.py"
    spec = importlib.util.spec_from_file_location("pes_scan_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rendered = module.YAML_TEMPLATE.format(distance=0.74804)
    parsed = yaml.safe_load(rendered)
    cfg = TopLevelConfig.parse(parsed)
    assert cfg.method.kind == "vqe"
    assert cfg.method.ansatz.kind == "ucc"
    # Sanity: distance interpolation actually landed in the geometry.
    assert any(abs(coord[2] - 0.74804) < 1e-6 for _sym, coord in parsed["molecule"]["geometry"])
