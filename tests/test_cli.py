"""End-to-end tests for the qibochem CLI."""

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

VQE_FAST_YAML = """\
version: 1
molecule:
  geometry:
    - [H, [0.0, 0.0, 0.0]]
    - [H, [0.0, 0.0, 0.74804]]
  basis: sto-3g
hamiltonian:
  mapping: jw
method:
  kind: vqe
  ansatz:
    kind: he
    layers: 1
  optimizer:
    name: BFGS
    options: {maxiter: 2}
    initial_parameters: zeros
    seed: 0
output:
  log_level: warning
"""


def _write(tmp_path: Path, text: str, name: str = "input.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_validate_ok(tmp_path: Path, capsys):
    yaml_path = _write(tmp_path, HF_YAML)
    rc = cli_main(["validate", str(yaml_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_validate_bad_yaml_returns_2(tmp_path: Path, capsys):
    bad = _write(tmp_path, "version: 1\nmolecule: {}\nhamiltonian: {}\nmethod: {kind: hf}\noutput: {}\n", "bad.yaml")
    rc = cli_main(["validate", str(bad)])
    assert rc == 2  # ConfigError -> EXIT_VALIDATION


def test_template_hf_emits_valid_yaml(tmp_path: Path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["template", "hf"])
    assert rc == 0
    text = buf.getvalue()
    p = _write(tmp_path, text, "from_template.yaml")
    assert cli_main(["validate", str(p)]) == 0


def test_run_hf_matches_pyscf(tmp_path: Path):
    yaml_path = _write(tmp_path, HF_YAML)
    rc = cli_main(["run", str(yaml_path)])
    assert rc == 0
    result = json.loads((tmp_path / "input.result.json").read_text())
    # H2/STO-3G HF energy reference (matches PySCF to all printed digits)
    assert abs(result["results"]["e_hf"] - (-1.1162837362742928)) < 1e-8
    assert result["results"]["hamiltonian"]["n_qubits"] == 4
    assert result["input_sha256"]  # populated
    assert result["qibochem_version"]


def test_run_vqe_smoke(tmp_path: Path):
    """Tiny VQE; just verify the pipeline runs and writes a parameters block."""
    yaml_path = _write(tmp_path, VQE_FAST_YAML)
    rc = cli_main(["run", str(yaml_path)])
    assert rc == 0
    result = json.loads((tmp_path / "input.result.json").read_text())
    assert "e_vqe" in result["results"]
    assert "final_parameters" in result["results"]
    assert result["results"]["ansatz"]["kind"] == "he"


def test_run_with_dry_run(tmp_path: Path, capsys):
    yaml_path = _write(tmp_path, HF_YAML)
    rc = cli_main(["run", str(yaml_path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["results"]["dry_run"] is True
    # Should NOT have written a result file in dry-run mode
    assert not (tmp_path / "input.result.json").exists()


def test_run_missing_file_returns_2(tmp_path: Path):
    rc = cli_main(["run", str(tmp_path / "nope.yaml")])
    assert rc == 2


def test_inspect_xyz(tmp_path: Path, capsys):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\n0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74804\n")
    rc = cli_main(["inspect", str(xyz)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "e_hf" in out
    assert "Pauli terms" in out


def test_subprocess_entry_point():
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
