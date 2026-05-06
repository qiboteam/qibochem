"""Schema-validation tests for the qibochem CLI input schema.

Most rejection tests are parametrised — one entry per invalid-input scenario,
all sharing the same fixture and assertion shape. Renaming an error message
only requires updating the relevant ``error_match`` regex once.
"""

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


# --------------------------------------------------------------------------- #
# Positive cases                                                              #
# --------------------------------------------------------------------------- #


def test_valid_hf_minimal():
    cfg = TopLevelConfig.parse(_hf_dict())
    assert isinstance(cfg.method, HFMethodConfig)
    assert cfg.molecule.basis == "sto-3g"
    assert cfg.hamiltonian.mapping == "jw"


def test_valid_vqe_full():
    cfg = TopLevelConfig.parse(_vqe_dict())
    assert isinstance(cfg.method, VQEMethodConfig)
    assert cfg.method.ansatz.kind == "ucc"
    assert cfg.method.optimizer.options["maxiter"] == 50


def test_xyz_xor_geometry_required():
    """Cross-field constraint: exactly one of xyz_file / geometry must be set."""
    d = _hf_dict()
    d["molecule"]["xyz_file"] = "h2.xyz"  # both set
    with pytest.raises(ConfigError, match="exactly one of"):
        TopLevelConfig.parse(d)
    d["molecule"].pop("geometry")
    d["molecule"].pop("xyz_file")  # neither set
    with pytest.raises(ConfigError, match="exactly one of"):
        TopLevelConfig.parse(d)


def test_template_yaml_files_parse():
    """Bundled CLI templates must validate against the schema they document."""
    from importlib import resources

    for kind in ("hf", "vqe"):
        text = resources.files("qibochem.cli.templates").joinpath(f"{kind}.yaml").read_text(encoding="utf-8")
        TopLevelConfig.parse(yaml.safe_load(text))  # raises if invalid


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
    assert any(abs(coord[2] - 0.74804) < 1e-6 for _sym, coord in parsed["molecule"]["geometry"])


# --------------------------------------------------------------------------- #
# Rejection cases — one parametrise per "type" of input                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mutator", "error_match"),
    [
        pytest.param(lambda d: d.update(version=2), "schema version", id="bad_version"),
        pytest.param(lambda d: d.update(mystery_field=42), "unknown keys", id="unknown_top_key"),
        pytest.param(lambda d: d["hamiltonian"].update(mapping="parity"), "hamiltonian.mapping", id="bad_mapping"),
        pytest.param(lambda d: d.__setitem__("method", {"kind": "monte-carlo"}), "method.kind", id="bad_method_kind"),
        pytest.param(lambda d: d.__setitem__("method", "hf"), "method: expected a mapping", id="method_not_dict"),
        pytest.param(
            lambda d: d.__setitem__("method", {"kind": "qse"}), "not available in this build", id="qse_phase3"
        ),
        pytest.param(
            lambda d: d.__setitem__("method", {"kind": "qsci"}), "not available in this build", id="qsci_phase3"
        ),
        pytest.param(lambda d: d["molecule"].update(multiplicity=0), "multiplicity", id="bad_multiplicity"),
        pytest.param(lambda d: d["molecule"].update(active="all"), "active must be a list of ints", id="bad_active"),
        pytest.param(lambda d: d["output"].update(log_level="trace"), "output.log_level", id="bad_log_level"),
        pytest.param(lambda d: d.update(version="one"), "version.*must be an integer", id="bad_version_unparseable"),
        pytest.param(lambda d: d["molecule"].update(charge=[1]), "charge.*must be an integer", id="bad_charge_list"),
    ],
)
def test_schema_rejects_invalid_top_level_inputs(mutator, error_match):
    d = _hf_dict()
    mutator(d)
    with pytest.raises(ConfigError, match=error_match):
        TopLevelConfig.parse(d)


@pytest.mark.parametrize(
    ("bad_geometry", "error_match"),
    [
        pytest.param([], "non-empty list", id="empty"),
        pytest.param([{"H": [0, 0, 0]}], r"expected \[symbol", id="wrong_atom_shape"),
        pytest.param([[1, [0.0, 0.0, 0.0]]], "atom symbol must be a string", id="non_string_symbol"),
        pytest.param([["H", [0.0, 0.0]]], "coordinates must be a 3-list", id="wrong_coord_dim"),
    ],
)
def test_schema_rejects_invalid_geometry(bad_geometry, error_match):
    d = _hf_dict()
    d["molecule"]["geometry"] = bad_geometry
    with pytest.raises(ConfigError, match=error_match):
        TopLevelConfig.parse(d)


@pytest.mark.parametrize(
    ("mutator", "error_match"),
    [
        pytest.param(lambda m: m["ansatz"].update(kind="transformer"), "method.ansatz.kind", id="bad_ansatz_kind"),
        pytest.param(lambda m: m["ansatz"].update(excitations=["triples"]), "invalid entries", id="bad_excitation"),
        pytest.param(lambda m: m["ansatz"].update(layers=0), "layers must be >= 1", id="bad_layers"),
        pytest.param(
            lambda m: m["optimizer"].update(options="maxiter=200"), "options must be a mapping", id="bad_options"
        ),
        pytest.param(
            lambda m: m["optimizer"].update(initial_parameters=[0.1, "oops"]),
            "must contain only numbers",
            id="bad_init_element",
        ),
        pytest.param(
            lambda m: m["optimizer"].update(initial_parameters="ones"), "initial_parameters", id="bad_init_strategy"
        ),
        pytest.param(
            lambda m: m["optimizer"].update(initial_parameters=42),
            "must be a string strategy or a list",
            id="bad_init_type",
        ),
        pytest.param(lambda m: m.update(n_shots=0), "n_shots", id="bad_n_shots_zero"),
        # Non-int values that can't be coerced must raise ConfigError (not a raw TypeError
        # that would surface as exit code 1).  Numeric-looking strings *are* accepted —
        # YAML treats `n_shots: "10"` and `n_shots: 10` interchangeably.
        pytest.param(lambda m: m.update(n_shots="ten"), "n_shots.*must be an integer", id="bad_n_shots_unparseable"),
        pytest.param(lambda m: m.update(n_shots=[100]), "n_shots.*must be an integer", id="bad_n_shots_list"),
        pytest.param(lambda m: m["ansatz"].update(layers=[2]), "layers.*must be an integer", id="bad_layers_list"),
        pytest.param(
            lambda m: m["optimizer"].update(seed="zero"),
            "method.optimizer.seed: must be an integer",
            id="bad_seed_unparseable",
        ),
    ],
)
def test_schema_rejects_invalid_vqe_inputs(mutator, error_match):
    d = _vqe_dict()
    mutator(d["method"])
    with pytest.raises(ConfigError, match=error_match):
        TopLevelConfig.parse(d)


def test_missing_required_key_message():
    """`_require` surfaces the key name and the block where it's missing."""
    with pytest.raises(ConfigError, match="missing required key 'method'"):
        TopLevelConfig.parse({"version": 1, "molecule": {"geometry": [["H", [0, 0, 0]]]}})


def test_top_level_must_be_mapping():
    with pytest.raises(ConfigError, match="top-level: expected a mapping"):
        TopLevelConfig.parse([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# I/O helpers (unit tests, no qibochem run)                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("yaml_text", "error_match"),
    [
        pytest.param("foo: [bar:\n", "YAML parse error", id="malformed"),
        pytest.param("", "empty", id="empty"),
        pytest.param("- foo\n- bar\n", "must be a mapping", id="non_mapping"),
    ],
)
def test_load_yaml_rejects_invalid(tmp_path: Path, yaml_text: str, error_match: str):
    from qibochem.cli.io import load_yaml

    p = tmp_path / "x.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ConfigError, match=error_match):
        load_yaml(p)


def test_resolve_xyz_path_handles_absolute_and_relative(tmp_path: Path):
    from qibochem.cli.io import resolve_xyz_path

    abs_xyz = (tmp_path / "thing.xyz").resolve()
    assert resolve_xyz_path(tmp_path / "input.yaml", str(abs_xyz)) == abs_xyz
    assert resolve_xyz_path(tmp_path / "input.yaml", "thing.xyz") == abs_xyz


def test_resolve_output_paths_absolute_dir(tmp_path: Path):
    from qibochem.cli.io import resolve_output_paths

    out_dir, prefix = resolve_output_paths(tmp_path / "x.yaml", str(tmp_path / "out"), None)
    assert out_dir == (tmp_path / "out").resolve()
    assert prefix == "x"


def test_serialise_handles_numpy_arrays_and_scalars():
    import numpy as np

    from qibochem.cli.io import _serialise

    out = _serialise({"arr": np.array([1.0, 2.0]), "scalar": np.float64(3.5), "n": np.int32(7)})
    assert out == {"arr": [1.0, 2.0], "scalar": 3.5, "n": 7}


def test_write_circuit_qasm_returns_none_when_unsupported(tmp_path: Path):
    from qibochem.cli.io import write_circuit_qasm

    assert write_circuit_qasm(object(), tmp_path, "test") is None
    assert not (tmp_path / "test.circuit.qasm").exists()


def test_git_commit_in_non_git_directory(tmp_path: Path, monkeypatch):
    from qibochem.cli.io import _git_commit

    monkeypatch.chdir(tmp_path)
    assert _git_commit() is None


def test_git_commit_when_git_binary_missing(monkeypatch):
    from qibochem.cli import io as io_mod

    def _raise(*_a, **_kw):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(io_mod.subprocess, "run", _raise)
    assert io_mod._git_commit() is None


# --------------------------------------------------------------------------- #
# Defensive helpers in runners.py                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("ham_factory", "expected"),
    [
        pytest.param(lambda: type("FakeHam", (), {})(), 0, id="no_form"),
        pytest.param(lambda: type("FakeHam", (), {"form": type("E", (), {"args": ()})()})(), 1, id="single_term"),
    ],
)
def test_n_pauli_terms(ham_factory, expected):
    from qibochem.cli.runners import _n_pauli_terms

    assert _n_pauli_terms(ham_factory()) == expected
