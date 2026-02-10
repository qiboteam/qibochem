"""Ground-state quantum-selected configuration interaction (QSCI)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from openfermion import QubitOperator
from qibo import gates


@dataclass(frozen=True)
class QSCIConfig:
    """Configuration for ground-state QSCI."""

    r: int
    n_roots: int | None = None
    n_shots: int | None = None
    n_electrons: int | None = None
    n_alpha: int | None = None
    n_beta: int | None = None
    postselect: bool = False
    seed: int | None = None

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError("`r` must be a positive integer.")
        if self.n_roots is not None and (self.n_roots <= 0 or self.n_roots > self.r):
            raise ValueError("`n_roots` must be a positive integer and cannot exceed `r`.")
        if self.n_shots is not None and self.n_shots <= 0:
            raise ValueError("`n_shots` must be a positive integer when provided.")
        if self.n_electrons is not None and self.n_electrons < 0:
            raise ValueError("`n_electrons` must be non-negative when provided.")
        if (self.n_alpha is None) != (self.n_beta is None):
            raise ValueError("`n_alpha` and `n_beta` must either both be set or both be None.")
        if self.n_alpha is not None and self.n_alpha < 0:
            raise ValueError("`n_alpha` must be non-negative when provided.")
        if self.n_beta is not None and self.n_beta < 0:
            raise ValueError("`n_beta` must be non-negative when provided.")
        if self.n_alpha is not None and self.n_electrons is not None:
            if self.n_alpha + self.n_beta != self.n_electrons:
                raise ValueError("`n_alpha + n_beta` must match `n_electrons` when all are provided.")
        if self.postselect and self.n_electrons is None and self.n_alpha is None:
            raise ValueError("`postselect=True` requires `n_electrons` and/or (`n_alpha`, `n_beta`).")


@dataclass(frozen=True)
class QSCIResult:
    """Subspace diagonalization output for ground-state QSCI."""

    energies: np.ndarray
    eigenvectors: np.ndarray
    selected_bitstrings: list[str]
    counts: dict[str, int]
    subspace_dimension: int
    n_shots_used: int

    @property
    def n_roots(self) -> int:
        """Number of eigenpairs returned."""
        return int(len(self.energies))

    @property
    def gs_energy(self) -> float:
        """Lowest-energy root in the selected subspace."""
        return float(np.real_if_close(self.energies[0]))

    @property
    def gs_coefficients(self) -> np.ndarray:
        """Coefficients of the lowest-energy root in the selected basis."""
        return self.eigenvectors[:, 0]

    def get_root(self, root_index: int) -> tuple[float, np.ndarray]:
        """Return ``(energy, coefficients)`` for the selected root index."""
        if root_index < 0 or root_index >= self.n_roots:
            raise IndexError(f"`root_index` must be in [0, {self.n_roots - 1}].")
        return float(np.real_if_close(self.energies[root_index])), self.eigenvectors[:, root_index]

    def coefficients_dict(self, root_index: int = 0) -> dict[str, complex]:
        """Return coefficients of one root as ``{bitstring: coefficient}``."""
        _, coeffs = self.get_root(root_index)
        return {bitstring: coeffs[i] for i, bitstring in enumerate(self.selected_bitstrings)}

    def degenerate_groups(self, atol: float = 1e-8, rtol: float = 1e-5) -> list[list[int]]:
        """Return index groups for numerically degenerate roots."""
        if self.n_roots < 2:
            return []

        groups = []
        group_start = 0
        for idx in range(1, self.n_roots):
            if not np.isclose(self.energies[idx], self.energies[idx - 1], atol=atol, rtol=rtol):
                if idx - group_start > 1:
                    groups.append(list(range(group_start, idx)))
                group_start = idx

        if self.n_roots - group_start > 1:
            groups.append(list(range(group_start, self.n_roots)))

        return groups


class QSCI:
    """Ground-state quantum-selected configuration interaction (QSCI)."""

    def __init__(self, qubit_hamiltonian: QubitOperator, n_qubits: int, config: QSCIConfig):
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("`qubit_hamiltonian` must be an `openfermion.QubitOperator`.")
        if n_qubits <= 0:
            raise ValueError("`n_qubits` must be a positive integer.")
        self.qubit_hamiltonian = qubit_hamiltonian
        self.n_qubits = n_qubits
        self.config = config

    def run_from_samples(self, samples: list[str]) -> QSCIResult:
        """Run ground-state QSCI from a list of measured bitstrings."""

        counts = _count_samples(samples, self.n_qubits)
        return self._run_from_counts(counts, n_shots_used=len(samples))

    def run_from_circuit(self, circuit: Any, n_shots: int | None = None) -> QSCIResult:
        """Run ground-state QSCI by sampling a Qibo circuit in the computational basis."""

        nshots = self.config.n_shots if n_shots is None else n_shots
        if nshots is None or nshots <= 0:
            raise ValueError("`n_shots` must be provided either in config or method call.")

        sampled_circuit = circuit.copy()
        if not sampled_circuit.measurements:
            sampled_circuit.add(gates.M(*range(self.n_qubits)))

        if self.config.seed is not None:
            random_state = np.random.get_state()
            np.random.seed(self.config.seed)
            try:
                frequencies = sampled_circuit(nshots=nshots).frequencies(binary=True)
            finally:
                np.random.set_state(random_state)
        else:
            frequencies = sampled_circuit(nshots=nshots).frequencies(binary=True)

        counts = _validate_and_copy_counts(dict(frequencies), self.n_qubits)
        return self._run_from_counts(counts, n_shots_used=nshots)

    def _run_from_counts(self, counts: dict[str, int], n_shots_used: int) -> QSCIResult:
        working_counts = dict(counts)
        if self.config.postselect:
            working_counts = _postselect_counts(
                working_counts,
                n_electrons=self.config.n_electrons,
                n_alpha=self.config.n_alpha,
                n_beta=self.config.n_beta,
            )

        if not working_counts:
            raise ValueError("No valid bitstrings available after applying selection/post-selection.")

        selected = _select_top_r(working_counts, self.config.r)
        projected_hamiltonian = _project_hamiltonian_subspace(self.qubit_hamiltonian, selected)

        eigenvalues, eigenvectors = np.linalg.eigh(projected_hamiltonian)
        n_roots = len(selected) if self.config.n_roots is None else min(self.config.n_roots, len(selected))
        energies = np.real_if_close(eigenvalues[:n_roots])
        root_vectors = eigenvectors[:, :n_roots]

        return QSCIResult(
            energies=energies,
            eigenvectors=root_vectors,
            selected_bitstrings=selected,
            counts={bitstring: working_counts[bitstring] for bitstring in selected},
            subspace_dimension=len(selected),
            n_shots_used=n_shots_used,
        )


def qsci_ground_state(
    qubit_hamiltonian: QubitOperator,
    n_qubits: int,
    samples: list[str] | None = None,
    circuit: Any | None = None,
    config: QSCIConfig | None = None,
    **config_kwargs,
) -> QSCIResult:
    """One-shot convenience wrapper for ground-state QSCI."""

    if config is not None and config_kwargs:
        raise ValueError("Pass either `config` or `config_kwargs`, but not both.")
    if config is None:
        config = QSCIConfig(**config_kwargs)

    if (samples is None) == (circuit is None):
        raise ValueError("Provide exactly one of `samples` or `circuit`.")

    runner = QSCI(qubit_hamiltonian=qubit_hamiltonian, n_qubits=n_qubits, config=config)
    if samples is not None:
        return runner.run_from_samples(samples)
    return runner.run_from_circuit(circuit)


def _count_samples(samples: list[str], n_qubits: int) -> dict[str, int]:
    if not samples:
        raise ValueError("`samples` must contain at least one bitstring.")
    counts = Counter(samples)
    return _validate_and_copy_counts(dict(counts), n_qubits)


def _validate_and_copy_counts(counts: dict[str, int], n_qubits: int) -> dict[str, int]:
    validated = {}
    for bitstring, count in counts.items():
        if count <= 0:
            continue
        _validate_bitstring(bitstring, n_qubits)
        validated[bitstring] = int(count)
    return validated


def _validate_bitstring(bitstring: str, n_qubits: int):
    if len(bitstring) != n_qubits:
        raise ValueError(f"Invalid bitstring length: expected {n_qubits}, got {len(bitstring)} ({bitstring}).")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError(f"Bitstring must contain only 0/1 characters ({bitstring}).")


def _postselect_counts(
    counts: dict[str, int],
    n_electrons: int | None,
    n_alpha: int | None,
    n_beta: int | None,
) -> dict[str, int]:
    filtered = {}
    for bitstring, count in counts.items():
        if n_electrons is not None and bitstring.count("1") != n_electrons:
            continue
        if n_alpha is not None and n_beta is not None:
            n_alpha_x, n_beta_x = _spin_resolved_popcount(bitstring)
            if n_alpha_x != n_alpha or n_beta_x != n_beta:
                continue
        filtered[bitstring] = count
    return filtered


def _spin_resolved_popcount(bitstring: str) -> tuple[int, int]:
    """Returns (n_alpha, n_beta) assuming interleaved spin-orbital indexing."""

    n_alpha = sum(int(bitstring[i]) for i in range(0, len(bitstring), 2))
    n_beta = sum(int(bitstring[i]) for i in range(1, len(bitstring), 2))
    return n_alpha, n_beta


def _select_top_r(counts: dict[str, int], r: int) -> list[str]:
    sorted_by_freq = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [bitstring for bitstring, _ in sorted_by_freq[: min(r, len(sorted_by_freq))]]


def _project_hamiltonian_subspace(qubit_hamiltonian: QubitOperator, basis_bitstrings: list[str]) -> np.ndarray:
    dim = len(basis_bitstrings)
    matrix = np.zeros((dim, dim), dtype=complex)
    basis_index = {bitstring: idx for idx, bitstring in enumerate(basis_bitstrings)}

    for col, ket in enumerate(basis_bitstrings):
        for pauli_term, coeff in qubit_hamiltonian.terms.items():
            bra, phase = _apply_pauli_term(pauli_term, ket)
            row = basis_index.get(bra)
            if row is not None:
                matrix[row, col] += coeff * phase

    matrix = 0.5 * (matrix + matrix.conj().T)
    if np.allclose(matrix.imag, 0.0, atol=1e-12):
        matrix = matrix.real
    return matrix


def _apply_pauli_term(pauli_term: tuple[tuple[int, str], ...], bitstring: str) -> tuple[str, complex]:
    bits = list(bitstring)
    phase = 1.0 + 0.0j

    for qubit, operator in pauli_term:
        bit = bits[qubit]
        if operator == "X":
            bits[qubit] = "1" if bit == "0" else "0"
        elif operator == "Y":
            phase *= 1.0j if bit == "0" else -1.0j
            bits[qubit] = "1" if bit == "0" else "0"
        elif operator == "Z":
            phase *= 1.0 if bit == "0" else -1.0
        else:
            raise ValueError(f"Unsupported Pauli operator `{operator}` in term `{pauli_term}`.")

    return "".join(bits), phase
