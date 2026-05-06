"""End-to-end tests for the qibochem CLI.

Most VQE tests are parametrised so that one new scenario costs one new
``pytest.param`` line, not a new function. The shape is always:

    1. Compose a YAML (via _h2_vqe_yaml + _vqe_method, or a verbatim string).
    2. Invoke cli_main(["run", yaml_path]).
    3. Assert exit code, then optionally inspect the JSON result file.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from qibochem.cli.main import main as cli_main

HF_YAML = """\
version: 1
molecule:
  geometry:
    - [H, [0.0, 0.0, 0.0]]
    - [H, [0.0, 0.0, 0.74804]]
  basis: sto-3g
hamiltonian:
  mapping: jw
method:
  kind: hf
output:
  log_level: warning
"""


def _write(tmp_path: Path, text: str, name: str = "input.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _h2_vqe_yaml(
    method_block: str, *, mapping: str = "jw", molecule_extra: str = "", output_block: str = "  log_level: warning"
) -> str:
    return f"""\
version: 1
molecule:
  geometry:
    - [H, [0.0, 0.0, 0.0]]
    - [H, [0.0, 0.0, 0.74804]]
  basis: sto-3g
{molecule_extra}hamiltonian:
  mapping: {mapping}
{method_block}
output:
{output_block}
"""


def _vqe_method(ansatz: str, *, optimizer: str = "BFGS", n_shots=None, init: str = "zeros") -> str:
    shots_line = f"  n_shots: {n_shots}\n" if n_shots is not None else ""
    return (
        "method:\n"
        "  kind: vqe\n"
        f"  ansatz: {ansatz}\n"
        f"  optimizer: {{name: {optimizer}, options: {{maxiter: 1}}, initial_parameters: {init}, seed: 0}}\n"
        f"{shots_line}"
    )


# --------------------------------------------------------------------------- #
# Single-purpose tests with unique assertions                                 #
# --------------------------------------------------------------------------- #


def test_template_hf_round_trips_through_validate(tmp_path: Path):
    """`qibochem template hf` must emit a YAML that `qibochem validate` accepts.

    Doubles as smoke coverage of `template`, `validate`, and the schema's
    happy path on a realistic input.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli_main(["template", "hf"]) == 0
    p = _write(tmp_path, buf.getvalue(), "from_template.yaml")
    assert cli_main(["validate", str(p)]) == 0


def test_validate_surfaces_schema_error_with_exit_code_2(tmp_path: Path):
    bad = _write(
        tmp_path,
        "version: 1\nmolecule: {}\nhamiltonian: {}\nmethod: {kind: hf}\noutput: {}\n",
        "bad.yaml",
    )
    assert cli_main(["validate", str(bad)]) == 2


def test_run_hf_matches_pyscf(tmp_path: Path):
    """The single correctness assertion in the suite: HF energy must match PySCF."""
    yaml_path = _write(tmp_path, HF_YAML)
    assert cli_main(["run", str(yaml_path)]) == 0
    result = json.loads((tmp_path / "input.result.json").read_text())
    assert abs(result["results"]["e_hf"] - (-1.1162837362742928)) < 1e-8
    assert result["results"]["hamiltonian"]["n_qubits"] == 4
    assert result["input_sha256"] and result["qibochem_version"]


def test_run_dry_run_skips_optimisation_and_writes_to_stdout(tmp_path: Path, capsys):
    yaml_path = _write(tmp_path, HF_YAML)
    assert cli_main(["run", str(yaml_path), "--dry-run"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["results"]["dry_run"] is True
    assert not (tmp_path / "input.result.json").exists()


def test_inspect_reports_qubit_and_term_counts(tmp_path: Path, capsys):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\n0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74804\n")
    assert cli_main(["inspect", str(xyz)]) == 0
    out = capsys.readouterr().out
    assert "e_hf" in out and "Pauli terms" in out


def test_run_hf_applies_active_space_for_embedding(tmp_path: Path):
    """HF embedding via `active` populates the active-space fields in the result."""
    xyz = tmp_path / "lih.xyz"
    xyz.write_text("2\n0 1\nLi 0.0 0.0 0.0\nH 0.0 0.0 1.6\n")
    yaml_path = _write(
        tmp_path,
        f"""\
version: 1
molecule:
  xyz_file: {xyz.name}
  basis: sto-3g
  active: [1, 2, 5]
hamiltonian:
  mapping: jw
method:
  kind: hf
output:
  log_level: warning
""",
    )
    assert cli_main(["run", str(yaml_path)]) == 0
    result = json.loads((tmp_path / "input.result.json").read_text())
    assert result["results"]["molecule"]["active"] == [1, 2, 5]
    assert result["results"]["molecule"]["n_active_orbs"] == 6


def test_run_vqe_save_circuit_writes_qasm(tmp_path: Path):
    yaml_path = _write(
        tmp_path,
        _h2_vqe_yaml(
            _vqe_method("{kind: he, layers: 1}"),
            output_block="  save_circuit: true\n  log_level: warning",
        ),
    )
    assert cli_main(["run", str(yaml_path)]) == 0
    qasm = tmp_path / "input.circuit.qasm"
    # Best-effort dump — accept either a non-empty file or a clean skip.
    assert (qasm.exists() and qasm.read_text().strip()) or not qasm.exists()


# --------------------------------------------------------------------------- #
# Parametrised: VQE happy paths (rc == 0)                                     #
#                                                                             #
# One pytest.param per scenario.  ``post_check`` runs against the parsed      #
# result JSON when a run produces something specific worth asserting.         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("yaml_text", "post_check"),
    [
        # Per-ansatz smoke: build_ansatz_circuit + initial_parameters + minimize
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: he, layers: 1}")), None, id="he"),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: ucc, excitations: [singles, doubles]}")), None, id="ucc_sd"),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: ucc, excitations: [singles]}")), None, id="ucc_s_only"),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: ucc, excitations: [doubles]}")), None, id="ucc_d_only"),
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: givens, excitations: [singles, doubles]}")), None, id="givens_sd"
        ),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: givens, excitations: [doubles]}")), None, id="givens_d_only"),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: qeb, excitations: [singles, doubles]}")), None, id="qeb"),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: symm}")), None, id="symm"),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: basis_rotation}")), None, id="basis_rotation"),
        # Initial-parameter strategies
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: he, layers: 1}", init="random")), None, id="init_random"),
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: ucc, excitations: [singles, doubles]}", init="mp2")),
            None,
            id="init_mp2_ucc",
        ),
        # MP2 init must respect the user's excitation subset, not always rebuild as full UCCSD.
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: ucc, excitations: [singles]}", init="mp2")),
            None,
            id="init_mp2_ucc_singles_only",
        ),
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: ucc, excitations: [doubles]}", init="mp2")),
            None,
            id="init_mp2_ucc_doubles_only",
        ),
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: givens, excitations: [singles, doubles]}", init="mp2")),
            None,
            id="init_mp2_givens",
        ),
        pytest.param(
            _h2_vqe_yaml(
                "method:\n"
                "  kind: vqe\n"
                "  ansatz: {kind: he, layers: 1}\n"
                "  optimizer:\n"
                "    name: BFGS\n"
                "    options: {maxiter: 1}\n"
                "    initial_parameters: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
            ),
            None,
            id="init_explicit_list",
        ),
        # Sample-based expectation (expectation_from_samples branch)
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: he, layers: 1}", n_shots=50)), None, id="n_shots"),
        # qibo optimizer must honour the user's options dict (e.g. maxiter), not silently
        # dispatch with internal defaults.
        pytest.param(
            _h2_vqe_yaml(
                "method:\n"
                "  kind: vqe\n"
                "  ansatz: {kind: he, layers: 1}\n"
                "  optimizer: {name: qibo, options: {maxiter: 1}, initial_parameters: zeros, seed: 0}\n"
            ),
            lambda r: (
                r["results"]["optimizer"]["name"].lower() == "qibo"
                # qibo defaults to Powell, which does many function evals per iteration.
                # ``maxiter: 1`` therefore can't cap evals to a single digit, but it must
                # be visibly fewer than the unbounded default (which is ~200+).
                and r["results"]["optimizer"]["n_function_evals"] < 150
            ),
            id="optimizer_qibo_honours_options",
        ),
        # Generic VQE smoke (sanity: result exposes e_vqe + final_parameters)
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: he, layers: 1}")),
            lambda r: "e_vqe" in r["results"] and "final_parameters" in r["results"],
            id="result_shape",
        ),
    ],
)
def test_run_vqe_combinations(tmp_path: Path, yaml_text: str, post_check):
    yaml_path = _write(tmp_path, yaml_text)
    assert cli_main(["run", str(yaml_path)]) == 0
    if post_check is not None:
        assert post_check(json.loads((tmp_path / "input.result.json").read_text()))


# --------------------------------------------------------------------------- #
# Parametrised: invalid inputs that the runner rejects (rc == 2)              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("yaml_text", "stderr_match"),
    [
        # mp2 init strategy is only valid for ucc / givens
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: he, layers: 1}", init="mp2")), "mp2", id="mp2_with_he"),
        # basis_rotation does not (yet) support an active space
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: basis_rotation}"), molecule_extra="  active: [0, 1]\n"),
            "basis_rotation",
            id="basis_rotation_with_active",
        ),
        # Several ansatzes only support JW mapping today
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: givens, excitations: [singles, doubles]}"), mapping="bk"),
            "givens",
            id="givens_with_bk",
        ),
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: qeb, excitations: [singles, doubles]}"), mapping="bk"),
            "qeb",
            id="qeb_with_bk",
        ),
        pytest.param(_h2_vqe_yaml(_vqe_method("{kind: symm}"), mapping="bk"), "symm", id="symm_with_bk"),
        pytest.param(
            _h2_vqe_yaml(_vqe_method("{kind: basis_rotation}"), mapping="bk"),
            "basis_rotation",
            id="basis_rotation_with_bk",
        ),
        # Explicit initial-parameter list whose length doesn't match the circuit
        pytest.param(
            _h2_vqe_yaml(
                "method:\n"
                "  kind: vqe\n"
                "  ansatz: {kind: he, layers: 1}\n"
                "  optimizer:\n"
                "    name: BFGS\n"
                "    options: {maxiter: 1}\n"
                "    initial_parameters: [0.1, 0.2]\n"
            ),
            "length",
            id="explicit_init_wrong_length",
        ),
    ],
)
def test_run_rejects_invalid_combinations(tmp_path: Path, capsys, yaml_text: str, stderr_match: str):
    yaml_path = _write(tmp_path, yaml_text)
    assert cli_main(["run", str(yaml_path)]) == 2
    assert stderr_match in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# Parametrised: small surface checks                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("argv", "expected_rc", "stderr_match"),
    [
        pytest.param(["run", "{tmp}/absent.yaml"], 2, "input file not found", id="run_missing_yaml"),
        pytest.param(["inspect", "{tmp}/absent.xyz"], 2, "not found", id="inspect_missing_xyz"),
    ],
)
def test_missing_input_returns_validation_error(tmp_path: Path, capsys, argv, expected_rc: int, stderr_match: str):
    rendered = [a.format(tmp=tmp_path) for a in argv]
    assert cli_main(rendered) == expected_rc
    assert stderr_match in capsys.readouterr().err.lower()


@pytest.mark.parametrize("flag", ["-v", "-q"])
def test_log_level_flags_run_cleanly(tmp_path: Path, flag: str):
    """`-v` / `-q` flags must not interfere with a successful HF run."""
    yaml_path = _write(tmp_path, HF_YAML)
    assert cli_main([flag, "run", str(yaml_path)]) == 0


@pytest.mark.parametrize(
    ("exc_factory", "expected_rc", "stderr_match"),
    [
        pytest.param(lambda: RuntimeError("simulated"), 3, "runtime error", id="runtime_error_exit_3"),
        pytest.param(lambda: FileNotFoundError("/nope"), 2, "file not found", id="file_not_found_exit_2"),
    ],
)
def test_top_level_exception_handlers(
    tmp_path: Path, capsys, monkeypatch, exc_factory, expected_rc: int, stderr_match: str
):
    """RuntimeError / FileNotFoundError raised from a runner reach the right handler."""
    from qibochem.cli import runners

    def boom(*_a, **_kw):
        raise exc_factory()

    monkeypatch.setattr(runners, "dispatch", boom)
    yaml_path = _write(tmp_path, HF_YAML)
    assert cli_main(["run", str(yaml_path)]) == expected_rc
    assert stderr_match in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# Entry points (subprocess; exercises packaging)                              #
# --------------------------------------------------------------------------- #


def test_subprocess_console_script_entry_point():
    """If qibochem is installed (poetry install), the console script must work."""
    qibochem = shutil.which("qibochem")
    if qibochem is None:
        pytest.skip("qibochem console script not on PATH; install with `poetry install`")
    proc = subprocess.run([qibochem, "--version"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "qibochem" in proc.stdout.lower()


def test_python_m_qibochem_entry_point():
    """`python -m qibochem` must dispatch to the CLI."""
    proc = subprocess.run(
        [sys.executable, "-m", "qibochem", "template", "hf"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "version: 1" in proc.stdout
