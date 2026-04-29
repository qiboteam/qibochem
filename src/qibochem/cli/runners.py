"""Method runners that translate validated CLI configs into Qibochem API calls.

These functions are deliberately thin: they build a ``Molecule``, a Hamiltonian,
and a circuit using the same public APIs a notebook user would. The CLI is a
YAML→Python-API translator — no method logic lives here.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qibochem.ansatz import (
    basis_rotation_gates,
    givens_excitation_ansatz,
    he_circuit,
    hf_circuit,
    qeb_circuit,
    symm_preserving_circuit,
    ucc_ansatz,
)
from qibochem.ansatz.basis_rotation import (
    basis_rotation_layout,
    givens_qr_decompose,
    unitary,
)
from qibochem.ansatz.util import generate_excitations, sort_excitations
from qibochem.cli.io import resolve_xyz_path
from qibochem.cli.schema import (
    ConfigError,
    HFMethodConfig,
    MoleculeConfig,
    OptimizerConfig,
    TopLevelConfig,
    VQEMethodConfig,
)
from qibochem.driver import Molecule
from qibochem.measurement import expectation, expectation_from_samples

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Molecule + Hamiltonian construction                                         #
# --------------------------------------------------------------------------- #


def build_molecule(cfg: MoleculeConfig, yaml_path: Path) -> Molecule:
    """Build and PySCF-populate a ``Molecule`` from the molecule config."""
    if cfg.xyz_file is not None:
        xyz = str(resolve_xyz_path(yaml_path, cfg.xyz_file))
        mol = Molecule(xyz_file=xyz, charge=cfg.charge, multiplicity=cfg.multiplicity, basis=cfg.basis)
    else:
        geometry = [(sym, tuple(xyz)) for sym, xyz in cfg.geometry]
        mol = Molecule(
            geometry=geometry,
            charge=cfg.charge,
            multiplicity=cfg.multiplicity,
            basis=cfg.basis,
        )
    mol.run_pyscf()
    if cfg.active is not None:
        mol.hf_embedding(active=list(cfg.active))
    return mol


def _molecular_formula(geometry) -> str:
    """Build a Hill-style molecular formula like 'H2', 'LiH', 'CH4'."""
    counts = Counter(sym for sym, _ in geometry)
    parts = []
    # Hill convention: C first, then H, then everything else alphabetical
    for sym in sorted(counts, key=lambda s: (s != "C", s != "H", s)):
        n = counts[sym]
        parts.append(f"{sym}{n}" if n > 1 else sym)
    return "".join(parts)


def molecule_summary(mol: Molecule) -> dict:
    return {
        "formula": _molecular_formula(mol.geometry),
        "n_atoms": len(mol.geometry),
        "nelec": int(mol.nelec),
        "nso": int(mol.nso),
        "active": list(mol.active) if mol.active is not None else None,
        "n_active_e": int(mol.n_active_e) if mol.n_active_e is not None else None,
        "n_active_orbs": int(mol.n_active_orbs) if mol.n_active_orbs is not None else None,
    }


def build_hamiltonian(mol: Molecule, mapping: str):
    return mol.hamiltonian("sym", ferm_qubit_map=mapping)


def _n_pauli_terms(ham) -> int:
    """Count Pauli terms in a Qibo SymbolicHamiltonian via its sympy ``form``."""
    form = getattr(ham, "form", None)
    if form is None:
        return 0
    args = getattr(form, "args", None)
    if not args:
        return 1
    # Add expression: each arg is one Pauli term
    return len(args)


# --------------------------------------------------------------------------- #
# Ansatz factories                                                            #
# --------------------------------------------------------------------------- #


def _active_space_dims(mol: Molecule) -> tuple[int, int]:
    n_elec = mol.nelec if mol.n_active_e is None else mol.n_active_e
    n_orbs = mol.nso if mol.n_active_orbs is None else mol.n_active_orbs
    return n_elec, n_orbs


def _excitation_orders(excitations: list[str]) -> list[int]:
    out = []
    if "singles" in excitations:
        out.append(1)
    if "doubles" in excitations:
        out.append(2)
    return sorted(out, reverse=True)  # higher order first, matching ucc_ansatz convention


def _gen_excitations(mol: Molecule, orders: list[int]) -> list[list[int]]:
    n_elec, n_orbs = _active_space_dims(mol)
    excs: list[list[int]] = []
    for order in orders:
        excs += sort_excitations(generate_excitations(order, range(0, n_elec), range(n_elec, n_orbs)))
    return excs


def _build_ucc(mol: Molecule, excitations: list[str], mapping: str):
    """UCC: full UCCSD by default, or a subset depending on ``excitations``."""
    orders = _excitation_orders(excitations)
    if orders == [2, 1]:
        excitation_level = "D"  # ucc_ansatz with D includes both S and D
        circuit = ucc_ansatz(mol, excitation_level=excitation_level, ferm_qubit_map=mapping, use_mp2_guess=False)
    elif orders == [1]:
        circuit = ucc_ansatz(mol, excitation_level="S", ferm_qubit_map=mapping, use_mp2_guess=False)
    elif orders == [2]:
        # Doubles only — pass explicit list
        excs = _gen_excitations(mol, [2])
        circuit = ucc_ansatz(mol, excitations=excs, ferm_qubit_map=mapping, use_mp2_guess=False)
    else:  # pragma: no cover - schema rules this out
        raise ConfigError("UCC requires at least one of singles/doubles")
    return circuit


def _build_givens(mol: Molecule, excitations: list[str]):
    orders = _excitation_orders(excitations)
    if orders == [2, 1]:
        return givens_excitation_ansatz(mol, use_mp2_guess=False)
    excs = _gen_excitations(mol, orders)
    return givens_excitation_ansatz(mol, excitations=excs, use_mp2_guess=False)


def _build_qeb(mol: Molecule, excitations: list[str]):
    """QEB: HF reference + qeb_circuit per excitation (mirrors ucc_example pattern)."""
    n_elec, n_orbs = _active_space_dims(mol)
    excs = _gen_excitations(mol, _excitation_orders(excitations))
    circuit = hf_circuit(n_orbs, n_elec)
    for ex in excs:
        circuit += qeb_circuit(n_orbs, ex)
    return circuit


def _build_he(mol: Molecule, layers: int):
    return he_circuit(int(mol.nso if mol.n_active_orbs is None else mol.n_active_orbs), layers)


def _build_symm(mol: Molecule):
    n_elec, n_orbs = _active_space_dims(mol)
    return symm_preserving_circuit(n_orbs, n_elec)


def _build_basis_rotation(mol: Molecule):
    """Mirrors examples/br_example.py: HF reference + Givens-QR rotation gates."""
    if mol.n_active_orbs is not None:
        raise ConfigError("ansatz.kind=basis_rotation does not currently support an active space")
    nqubits = mol.nso
    occ = range(0, mol.nelec)
    vir = range(mol.nelec, mol.nso)
    U, theta = unitary(occ, vir, parameters=0.0)
    gate_angles, _ = givens_qr_decompose(U)
    layout = basis_rotation_layout(nqubits)
    gates, _ = basis_rotation_gates(layout, gate_angles, theta)
    circuit = hf_circuit(nqubits, mol.nelec)
    circuit.add(gates)
    return circuit


def build_ansatz_circuit(mol: Molecule, ansatz_cfg, mapping: str):
    kind = ansatz_cfg.kind
    if kind == "ucc":
        return _build_ucc(mol, ansatz_cfg.excitations, mapping)
    if kind == "givens":
        if mapping != "jw":
            raise ConfigError("ansatz.kind=givens currently only supports JW mapping")
        return _build_givens(mol, ansatz_cfg.excitations)
    if kind == "qeb":
        if mapping != "jw":
            raise ConfigError("ansatz.kind=qeb currently only supports JW mapping")
        return _build_qeb(mol, ansatz_cfg.excitations)
    if kind == "he":
        return _build_he(mol, ansatz_cfg.layers)
    if kind == "symm":
        if mapping != "jw":
            raise ConfigError("ansatz.kind=symm currently only supports JW mapping")
        return _build_symm(mol)
    if kind == "basis_rotation":
        if mapping != "jw":
            raise ConfigError("ansatz.kind=basis_rotation currently only supports JW mapping")
        return _build_basis_rotation(mol)
    raise ConfigError(f"unknown ansatz kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Initial parameters                                                          #
# --------------------------------------------------------------------------- #


def _flatten_circuit_params(circuit) -> np.ndarray:
    """Flatten ``circuit.get_parameters()`` into a real-valued 1-D float array.

    Qibo returns parameters as a list of per-gate tuples whose entries may be
    complex (e.g. UCC's MP2 guess has tiny imaginary parts). We take the real
    part — the imaginary component is numerical noise on a unitary phase.
    """
    raw = circuit.get_parameters()
    flat: list[float] = []
    for entry in raw:
        if isinstance(entry, (list, tuple, np.ndarray)):
            for x in entry:
                flat.append(float(np.real(x)))
        else:
            flat.append(float(np.real(entry)))
    return np.asarray(flat, dtype=float)


def _n_circuit_params(circuit) -> int:
    try:
        return int(len(_flatten_circuit_params(circuit)))
    except Exception:
        return 0


def initial_parameters(circuit, opt_cfg: OptimizerConfig, mol: Molecule, ansatz_cfg) -> np.ndarray:
    n = _n_circuit_params(circuit)
    if n == 0:
        return np.zeros(0)

    strategy = opt_cfg.initial_parameters
    if isinstance(strategy, list):
        if len(strategy) != n:
            raise ConfigError(f"optimizer.initial_parameters has length {len(strategy)} but circuit expects {n}")
        return np.asarray(strategy, dtype=float)

    if strategy == "zeros":
        return np.zeros(n)

    if strategy == "random":
        rng = np.random.default_rng(opt_cfg.seed)
        return rng.uniform(-np.pi, np.pi, size=n)

    if strategy == "mp2":
        if ansatz_cfg.kind not in ("ucc", "givens"):
            raise ConfigError(
                f"initial_parameters='mp2' is only supported for ansatz.kind in (ucc, givens); "
                f"got {ansatz_cfg.kind!r}"
            )
        if ansatz_cfg.kind == "ucc":
            mp2_circuit = ucc_ansatz(mol, excitation_level="D", use_mp2_guess=True)
        else:
            mp2_circuit = givens_excitation_ansatz(mol, use_mp2_guess=True)
        return _flatten_circuit_params(mp2_circuit)

    raise ConfigError(f"unknown initial_parameters strategy: {strategy!r}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Public runners                                                              #
# --------------------------------------------------------------------------- #


def run_hf(top: TopLevelConfig, yaml_path: Path) -> dict:
    """Hartree-Fock only: build molecule, run PySCF, report e_hf."""
    if not isinstance(top.method, HFMethodConfig):
        raise ConfigError("run_hf called with non-HF method config")
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    mol = build_molecule(top.molecule, yaml_path)
    timing["pyscf"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ham = build_hamiltonian(mol, top.hamiltonian.mapping)
    timing["hamiltonian"] = time.perf_counter() - t0

    results = {
        "molecule": molecule_summary(mol),
        "hamiltonian": {
            "mapping": top.hamiltonian.mapping,
            "n_qubits": int(mol.nso),
            "n_pauli_terms": _n_pauli_terms(ham),
        },
        "e_hf": float(mol.e_hf),
        "e_nuc": float(mol.e_nuc),
    }
    return {"results": results, "timing": timing, "circuit": None}


def run_vqe(top: TopLevelConfig, yaml_path: Path) -> dict:
    """VQE: build molecule + ansatz, optimize, report energies and parameters."""
    if not isinstance(top.method, VQEMethodConfig):
        raise ConfigError("run_vqe called with non-VQE method config")
    method: VQEMethodConfig = top.method
    timing: dict[str, float] = {}

    if method.optimizer.seed is not None:
        np.random.seed(method.optimizer.seed)
        random.seed(method.optimizer.seed)

    t0 = time.perf_counter()
    mol = build_molecule(top.molecule, yaml_path)
    timing["pyscf"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hamiltonian = build_hamiltonian(mol, top.hamiltonian.mapping)
    timing["hamiltonian"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    circuit = build_ansatz_circuit(mol, method.ansatz, top.hamiltonian.mapping)
    timing["ansatz_build"] = time.perf_counter() - t0

    params0 = initial_parameters(circuit, method.optimizer, mol, method.ansatz)
    n_params = len(params0)
    logger.info("VQE: %d parameters; ansatz=%s; optimizer=%s", n_params, method.ansatz.kind, method.optimizer.name)

    n_evals = {"count": 0}

    def loss(theta: np.ndarray) -> float:
        circuit.set_parameters(theta)
        n_evals["count"] += 1
        if method.n_shots is None:
            return float(expectation(circuit, hamiltonian))
        return float(expectation_from_samples(circuit, hamiltonian, n_shots=method.n_shots))

    t0 = time.perf_counter()
    if method.optimizer.name.lower() == "qibo":
        from qibo.optimizers import optimize as qibo_optimize

        best_energy, best_params, _extra = qibo_optimize(loss, params0)
        success = True
        message = "qibo.optimizers.optimize"
    else:
        result = minimize(loss, params0, method=method.optimizer.name, options=method.optimizer.options)
        best_energy = float(result.fun)
        best_params = np.asarray(result.x, dtype=float)
        success = bool(result.success)
        message = str(result.message)
    timing["optimization"] = time.perf_counter() - t0

    results = {
        "molecule": molecule_summary(mol),
        "hamiltonian": {
            "mapping": top.hamiltonian.mapping,
            "n_qubits": int(mol.nso),
            "n_pauli_terms": _n_pauli_terms(hamiltonian),
        },
        "ansatz": {"kind": method.ansatz.kind, "n_parameters": int(n_params)},
        "e_hf": float(mol.e_hf),
        "e_vqe": float(best_energy),
        "optimizer": {
            "name": method.optimizer.name,
            "n_function_evals": int(n_evals["count"]),
            "success": success,
            "message": message,
        },
    }
    if top.output.save_parameters:
        results["initial_parameters"] = params0.tolist()
        results["final_parameters"] = best_params.tolist()

    # Apply best params to the circuit so callers writing QASM see the optimised circuit.
    if n_params > 0:
        circuit.set_parameters(best_params)
    return {"results": results, "timing": timing, "circuit": circuit}


def dispatch(top: TopLevelConfig, yaml_path: Path) -> dict:
    """Run the configured method and return the runner result dict."""
    if isinstance(top.method, HFMethodConfig):
        return run_hf(top, yaml_path)
    if isinstance(top.method, VQEMethodConfig):
        return run_vqe(top, yaml_path)
    raise ConfigError(f"no runner for method type {type(top.method).__name__}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Inspect (no YAML — quick sanity check on an .xyz file)                      #
# --------------------------------------------------------------------------- #


def run_inspect(xyz_path: Path, basis: str = "sto-3g", mapping: str = "jw") -> dict:
    if not xyz_path.is_file():
        raise ConfigError(f"xyz file not found: {xyz_path}")
    mol = Molecule(xyz_file=str(xyz_path), basis=basis)
    mol.run_pyscf()
    ham = build_hamiltonian(mol, mapping)
    return {
        "xyz_file": str(xyz_path),
        "basis": basis,
        "mapping": mapping,
        "molecule": molecule_summary(mol),
        "e_hf": float(mol.e_hf),
        "e_nuc": float(mol.e_nuc),
        "n_qubits": int(mol.nso),
        "n_pauli_terms": _n_pauli_terms(ham),
    }


__all__ = ["dispatch", "run_hf", "run_inspect", "run_vqe"]
