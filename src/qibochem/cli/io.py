"""I/O helpers for the qibochem CLI: YAML loading, path resolution, result writing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from qibochem.cli.schema import ConfigError, TopLevelConfig

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    """Read and parse a YAML file using safe_load."""
    if not path.is_file():
        raise ConfigError(f"input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML parse error in {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"input file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(f"top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return data


def parse_config(path: Path) -> TopLevelConfig:
    """Load + validate a YAML input file into a ``TopLevelConfig``."""
    return TopLevelConfig.parse(load_yaml(path))


def resolve_xyz_path(yaml_path: Path, xyz_file: str) -> Path:
    """Paths in YAML are resolved relative to the YAML file's directory, not CWD."""
    p = Path(xyz_file)
    if not p.is_absolute():
        p = yaml_path.parent / p
    return p.resolve()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _qibochem_version() -> str:
    try:
        from qibochem import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _serialise(obj: Any) -> Any:
    """Recursively convert dataclasses/numpy arrays to JSON-friendly forms."""
    if is_dataclass(obj):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:  # pragma: no cover
        pass
    return obj


def build_result_envelope(
    *,
    input_path: Path,
    config: TopLevelConfig,
    results: dict,
    timing_s: dict,
) -> dict:
    """Assemble the JSON result envelope with reproducibility metadata."""
    return {
        "qibochem_version": _qibochem_version(),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "config": _serialise(config),
        "results": _serialise(results),
        "timing_s": {k: round(float(v), 4) for k, v in timing_s.items()},
    }


def write_result(envelope: dict, out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}.result.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, sort_keys=False)
        f.write("\n")
    return out_path


def write_circuit_qasm(circuit, out_dir: Path, prefix: str) -> Path | None:
    """Best-effort OpenQASM dump. Returns the path written, or None if unsupported."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}.circuit.qasm"
    to_qasm = getattr(circuit, "to_qasm", None)
    if to_qasm is None:
        logger.warning("circuit object has no to_qasm(); skipping QASM dump")
        return None
    try:
        text = to_qasm()
    except Exception as exc:  # pragma: no cover - depends on qibo internals
        logger.warning("to_qasm() failed (%s); skipping QASM dump", exc)
        return None
    out_path.write_text(text, encoding="utf-8")
    return out_path


def resolve_output_paths(yaml_path: Path, out_dir: str | None, prefix: str | None) -> tuple[Path, str]:
    """Return ``(out_dir, prefix)`` with defaults applied."""
    if out_dir is None:
        directory = yaml_path.parent
    else:
        d = Path(out_dir)
        directory = d if d.is_absolute() else (yaml_path.parent / d)
    if prefix is None:
        prefix = yaml_path.stem
    return directory.resolve(), prefix


def setup_logging(level: str) -> None:
    """Configure root logger for CLI output. Idempotent."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)


__all__ = [
    "build_result_envelope",
    "file_sha256",
    "load_yaml",
    "parse_config",
    "resolve_output_paths",
    "resolve_xyz_path",
    "setup_logging",
    "write_circuit_qasm",
    "write_result",
]


# Suppress noisy third-party loggers when imported as a module.
os.environ.setdefault("PYSCF_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
