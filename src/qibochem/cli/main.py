"""Argparse entry point for the ``qibochem`` CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from importlib import resources
from pathlib import Path

from qibochem.cli import runners
from qibochem.cli.io import (
    build_result_envelope,
    parse_config,
    resolve_output_paths,
    setup_logging,
    write_circuit_qasm,
    write_result,
)
from qibochem.cli.schema import ConfigError

logger = logging.getLogger("qibochem.cli")

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_VALIDATION = 2
EXIT_RUNTIME = 3

VALID_TEMPLATES = ("hf", "vqe")


def _qibochem_version() -> str:
    try:
        from qibochem import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qibochem",
        description="Run quantum-chemistry workflows declaratively from a YAML input file.",
    )
    parser.add_argument("--version", action="version", version=f"qibochem {_qibochem_version()}")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    log_group.add_argument("-q", "--quiet", action="store_true", help="Suppress info-level logging.")

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_run = sub.add_parser("run", help="Validate and execute a YAML input file.")
    p_run.add_argument("input", type=Path, help="Path to the YAML input file.")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, build the molecule and Hamiltonian, but skip optimisation. Useful for cost estimates.",
    )

    p_val = sub.add_parser("validate", help="Schema-validate a YAML input file without running.")
    p_val.add_argument("input", type=Path, help="Path to the YAML input file.")

    p_tpl = sub.add_parser("template", help="Print a starter YAML template to stdout.")
    p_tpl.add_argument("kind", choices=VALID_TEMPLATES, help="Template kind.")

    p_ins = sub.add_parser("inspect", help="Quick PySCF + Hamiltonian sanity check on an .xyz file.")
    p_ins.add_argument("xyz", type=Path, help="Path to a .xyz file.")
    p_ins.add_argument("--basis", default="sto-3g", help="AO basis set (default: sto-3g).")
    p_ins.add_argument("--mapping", default="jw", choices=["jw", "bk"], help="Fermion-to-qubit mapping.")

    return parser


# --------------------------------------------------------------------------- #
# Subcommand handlers                                                         #
# --------------------------------------------------------------------------- #


def _cmd_run(args: argparse.Namespace) -> int:
    yaml_path = args.input.resolve()
    config = parse_config(yaml_path)
    setup_logging(config.output.log_level if not (args.verbose or args.quiet) else _override_log(args))

    logger.info("Loaded config from %s (method=%s)", yaml_path, type(config.method).__name__)

    if args.dry_run:
        # Build molecule + hamiltonian only; skip optimisation
        mol = runners.build_molecule(config.molecule, yaml_path)
        ham = runners.build_hamiltonian(mol, config.hamiltonian.mapping)
        envelope = build_result_envelope(
            input_path=yaml_path,
            config=config,
            results={
                "dry_run": True,
                "molecule": runners.molecule_summary(mol),
                "hamiltonian": {
                    "mapping": config.hamiltonian.mapping,
                    "n_qubits": int(mol.nso),
                    "n_pauli_terms": runners._n_pauli_terms(ham),
                },
                "e_hf": float(mol.e_hf),
            },
            timing_s={"total": 0.0},
        )
        json.dump(envelope, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return EXIT_OK

    t0 = time.perf_counter()
    out = runners.dispatch(config, yaml_path)
    total = time.perf_counter() - t0
    timing = dict(out.get("timing", {}))
    timing["total"] = total

    out_dir, prefix = resolve_output_paths(yaml_path, config.output.dir, config.output.prefix)
    envelope = build_result_envelope(
        input_path=yaml_path,
        config=config,
        results=out["results"],
        timing_s=timing,
    )
    out_path = write_result(envelope, out_dir, prefix)
    logger.info("Wrote results to %s", out_path)

    circuit = out.get("circuit")
    if config.output.save_circuit and circuit is not None:
        qasm_path = write_circuit_qasm(circuit, out_dir, prefix)
        if qasm_path is not None:
            logger.info("Wrote circuit QASM to %s", qasm_path)

    _print_summary(out["results"], total)
    return EXIT_OK


def _override_log(args: argparse.Namespace) -> str:
    if args.verbose:
        return "debug"
    if args.quiet:
        return "warning"
    return "info"


def _cmd_validate(args: argparse.Namespace) -> int:
    yaml_path = args.input.resolve()
    parse_config(yaml_path)
    print(f"OK: {yaml_path} validates against the qibochem CLI schema (v1).")
    return EXIT_OK


def _cmd_template(args: argparse.Namespace) -> int:
    text = resources.files("qibochem.cli.templates").joinpath(f"{args.kind}.yaml").read_text(encoding="utf-8")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    xyz = args.xyz.resolve()
    info = runners.run_inspect(xyz, basis=args.basis, mapping=args.mapping)
    print(f"File:        {info['xyz_file']}")
    print(f"Basis:       {info['basis']}")
    print(f"Mapping:     {info['mapping']}")
    print(f"Atoms:       {len(info['molecule'].get('formula', '')) or '?'}  ({info['molecule']['formula']})")
    print(f"Electrons:   {info['molecule']['nelec']}")
    print(f"Spin orbs:   {info['molecule']['nso']} (= n_qubits)")
    print(f"e_nuc:       {info['e_nuc']:.8f} Ha")
    print(f"e_hf:        {info['e_hf']:.8f} Ha")
    print(f"Pauli terms: {info['n_pauli_terms']} ({info['mapping']} mapping)")
    return EXIT_OK


def _print_summary(results: dict, total_s: float) -> None:
    """Concise stdout summary alongside the JSON file."""
    print()
    print("=" * 60)
    if "e_hf" in results:
        print(f"  HF energy:  {results['e_hf']:.8f} Ha")
    if "e_vqe" in results:
        print(f"  VQE energy: {results['e_vqe']:.8f} Ha")
        opt = results.get("optimizer", {})
        print(f"    optimizer={opt.get('name')}  evals={opt.get('n_function_evals')}  ok={opt.get('success')}")
    print(f"  total wall time: {total_s:.2f} s")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Top-level dispatch                                                          #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Set early logging in case validate/template emit warnings
    setup_logging(_override_log(args))

    handlers = {
        "run": _cmd_run,
        "validate": _cmd_validate,
        "template": _cmd_template,
        "inspect": _cmd_inspect,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse already enforces
        parser.error(f"unknown command: {args.command}")

    try:
        return handler(args)
    except ConfigError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_VALIDATION
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: file not found: {exc}\n")
        return EXIT_VALIDATION
    except RuntimeError as exc:
        sys.stderr.write(f"runtime error: {exc}\n")
        return EXIT_RUNTIME
    except KeyboardInterrupt:  # pragma: no cover
        sys.stderr.write("interrupted\n")
        return EXIT_RUNTIME
    except Exception as exc:  # pragma: no cover - last-resort guard
        logger.exception("unexpected error")
        sys.stderr.write(f"unexpected error: {exc}\n")
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    raise SystemExit(main())
