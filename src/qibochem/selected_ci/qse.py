"""Quantum Subspace Expansion (QSE)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
import openfermion
from qibo.hamiltonians import SymbolicHamiltonian
from sympy import expand

from qibochem.driver.hamiltonian import (
    _qubit_hamiltonian,
    _qubit_to_symbolic_hamiltonian,
)
from qibochem.measurement.optimization import measurement_basis_rotations
from qibochem.measurement.shot_allocation import allocate_shots


@dataclass
class QSEConfig:
    """
    Configuration for Quantum Subspace Expansion (QSE).

    Args:
        conserve_spin (bool): Whether to only generate excitations that conserve spin.
        ferm_qubit_map (str): Mapping from fermions to qubits. Must be "jw" or "bk".
            WARNING: Ensure this matches the mapping used to construct the input circuit.
        excitation_threshold (float): Threshold for excitation norm to be included in the subspace.
        eigenvalue_threshold (float): Threshold for overlap matrix eigenvalues to avoid singular metric.
        n_shots (int): Optional number of shots to use for circuit measurements.
            WARNING: This defines the number of shots *per commuting Pauli group*.
            Total shots executed will be `n_shots * number_of_groups`. If None, exact statevector is used.
    """

    conserve_spin: bool = True
    ferm_qubit_map: str = "jw"
    excitation_threshold: float = 1e-9
    eigenvalue_threshold: float = 1e-9
    n_shots: int = None

    def __post_init__(self):
        if self.excitation_threshold < 0.0:
            raise ValueError("excitation_threshold must be non-negative.")
        if self.eigenvalue_threshold < 0.0:
            raise ValueError("eigenvalue_threshold must be non-negative.")
        if self.n_shots is not None and self.n_shots <= 0:
            raise ValueError("n_shots must be a positive integer.")
        if self.ferm_qubit_map not in ("jw", "bk"):
            raise ValueError("ferm_qubit_map must be 'jw' or 'bk'.")


@dataclass
class QSEResult:
    energies: np.ndarray
    eigenvectors: np.ndarray
    excitation_operators: list[openfermion.FermionOperator]
    subspace_dimension: int
    projected_subspace_dimension: int
    # total_circuit_runs: int = 0


class QSE:
    def __init__(self, molecule, n_shots=None, config=None):
        """
        Quantum Subspace Expansion (QSE) manager.

        Args:
            molecule: `qibochem.driver.Molecule` instance.
            config: Configuration for QSE.
        """
        self.molecule = molecule
        self.mol_hamiltonian = molecule.hamiltonian("f")  # Molecular Hamiltonian as FermionOperator
        self.config = config or QSEConfig()
        self.n_shots = n_shots
        operators = None  # List of fermion excitation operators
        self.dim = None
        self.s_data = None
        self.h_data = None
        self.S = None
        self.H = None
        self.term_exp_values = {}  # Expectation values for all terms

    def _generate_excitation_operators(
        self, nso: int, to_qubit_op: bool = False
    ) -> list[openfermion.FermionOperator | openfermion.QubitOperator]:
        """Generate one-body excitation operators + the identity."""
        # ZC note: Edited to match inquanto's singlet excitations (I think)
        operators = [
            openfermion.FermionOperator(f"{2*_i}^ {2*_j}") + openfermion.FermionOperator(f"{2*_i+1}^ {2*_j+1}")
            for _i in range(nso // 2)
            for _j in range(nso // 2)
        ]
        # operators = [openfermion.FermionOperator("")]
        # for i in range(nso):
        #     for j in range(nso):
        #         if self.config.conserve_spin and (i % 2) != (j % 2):
        #             continue
        #         operators.append(openfermion.FermionOperator(f"{i}^ {j}"))
        if to_qubit_op:
            operators = [self._fermion_to_qubit(op) for op in operators]
        return operators

    def collate_hs_matrix_info(self):
        """Constructs the Hamiltonian and terms for each of the S/H matrix elements"""
        # Determine number of active spin-orbitals (use active space if specified)
        n_active_orbs = self.molecule.n_active_orbs if self.molecule.n_active_orbs is not None else self.molecule.nso

        # Generate excitation operators
        self.operators = self._generate_excitation_operators(n_active_orbs)
        self.dim = len(self.operators)  # Subspace dimension

        # Store information about H/S in two dictionaries first
        self.s_data = {(_i, _j): dict() for _i in range(self.dim) for _j in range(self.dim) if _i <= _j}
        self.h_data = {(_i, _j): dict() for _i in range(self.dim) for _j in range(self.dim) if _i <= _j}
        # _qubit_to_symbolic_hamiltonian

        # Populate the Hamiltonians corresponding to each matrix element in S/H
        for mat_data, operator in zip((self.s_data, self.h_data), (1.0, self.mol_hamiltonian)):
            for element in mat_data.keys():
                mat_data[element]["ham"] = _qubit_to_symbolic_hamiltonian(
                    _qubit_hamiltonian(
                        openfermion.hermitian_conjugated(self.operators[element[0]])
                        * operator
                        * self.operators[element[1]],
                        self.config.ferm_qubit_map,
                    ),
                    self.dim,
                )
                mat_data[element]["terms"] = {
                    "".join(f"{string}{qubit}" for string, qubit in zip(strings, qubits)): coeff.real
                    for coeff, strings, qubits in zip(*mat_data[element]["ham"].simple_terms)
                }

    def _measure_term_group_shots(self, pauli_group, circuit, m_gates, n_shots):  # , return_metadata: bool = False):
        """Measure expectation values of a group of Pauli strings and optionally return measurement metadata."""
        from qibo.symbols import Z  # Why cannot use top-level import???

        _circuit = circuit.copy()
        _circuit.add(m_gates)
        result = _circuit(nshots=n_shots)
        frequencies = result.frequencies(binary=True)
        qubit_map = sorted(qubit for gate in m_gates for qubit in gate.target_qubits)
        if frequencies:  # Needed to handle n_shots = 0
            # Calculate expectation value for each term in the combined expression
            # Working with SymbolicHamiltonian directly at the moment, probably need to refactor old code (TODO)
            hamiltonian = SymbolicHamiltonian(pauli_group, nqubits=circuit.nqubits)
            # print(hamiltonian)
            for _coeff, strings, qubits in zip(*hamiltonian.simple_terms):
                term = "".join(f"{string}{qubit}" for string, qubit in zip(strings, qubits))
                # print(term)
                z_ham = SymbolicHamiltonian(prod(Z(qubit) for qubit in qubits), nqubits=circuit.nqubits)
                # print(z_ham)
                # print()
                self.term_exp_values[term] = z_ham.expectation_from_samples(frequencies, qubit_map=qubit_map)

    def calculate_hs_matrices(self, circuit, uniform_shot_allocation):
        """
        Calculates the H and S matrices
        """
        self.collate_hs_matrix_info()
        # print("s_data")
        # for entry, data in s_data.items():
        #     print(entry)
        #     print(data["ham"])
        #     print(data["ham"].expectation(circuit))
        #     print(data["terms"])
        #     print()

        # Now populate S/H matrices
        self.S = np.zeros((self.dim, self.dim), dtype=complex)
        self.H = np.zeros((self.dim, self.dim), dtype=complex)

        if self.n_shots is not None:
            # Combine all terms into a single Hamiltonian for shot allocation
            combined_ham = sum(
                (
                    data["ham"] if _i == _j else SymbolicHamiltonian(expand(2 * data["ham"].form))
                )  # Off-diagonal terms should be double counted
                for matrix in (self.s_data, self.h_data)
                for (_i, _j), data in matrix.items()
            )
            # print(combined_ham)
            # return

            grouped_terms = measurement_basis_rotations(combined_ham, grouping="qwc")

            if uniform_shot_allocation:
                # Get the expectation value for every term (without coefficients) based on the given shot allocation
                for pauli_group, m_gates in grouped_terms:
                    self._measure_term_group_shots(pauli_group, circuit, m_gates, self.n_shots)
            else:
                # Shot allocation based on the total magnitude of the coefficients in each group of terms ("qwc")
                shot_allocation = allocate_shots(grouped_terms, self.n_shots, method="c")
                # print(f"{shot_allocation = }")
                # Get the expectation value for every term (without coefficients) based on the given shot allocation
                for (pauli_group, m_gates), shots in zip(grouped_terms, shot_allocation):
                    self._measure_term_group_shots(pauli_group, circuit, m_gates, shots)

            # print(f"{self.term_exp_values = }")

            # Calculate H/S based on the Hamiltonian terms associated to each matrix element
            for _i in range(self.dim):
                for _j in range(self.dim):
                    if _i > _j:
                        # Populate with (_j, _i)
                        self.S[_i, _j] = np.conj(self.S[_j, _i])
                        self.H[_i, _j] = np.conj(self.H[_j, _i])
                        continue
                    self.S[_i, _j] = sum(
                        coeff * self.term_exp_values[term] for term, coeff in self.s_data[(_i, _j)]["terms"].items()
                    )
                    self.H[_i, _j] = sum(
                        coeff * self.term_exp_values[term] for term, coeff in self.h_data[(_i, _j)]["terms"].items()
                    )

        else:
            # Populate directly using the Hamiltonian for each matrix element
            for _i in range(self.dim):
                for _j in range(self.dim):
                    if _i > _j:
                        # Populate with (_j, _i)
                        self.S[_i, _j] = np.conj(self.S[_j, _i])
                        self.H[_i, _j] = np.conj(self.H[_j, _i])
                        continue
                    # print(_i, _j)
                    self.S[_i, _j] = self.s_data[(_i, _j)]["ham"].expectation(circuit)
                    self.H[_i, _j] = self.h_data[(_i, _j)]["ham"].expectation(circuit)

        # TODO: Can/Want to merge into existing code?
        # # Filter out operators that annihilate the state to avoid singular overlap matrix
        # valid_operators = []
        # valid_operators_qubit = []
        # for op, op_q, S_aa_q in zip(operators, operators_qubit, S_aa_qs):
        #     if self.config.n_shots is not None:
        #         norm_sq = np.real(sum(coeff * Saa_exp_vals[pauli] for pauli, coeff in S_aa_q.terms.items()))
        #     else:
        #         norm_sq = np.real(self._expectation_value_qubit(S_aa_q, statevector, n_active_orbs))

        #     if norm_sq > self.config.excitation_threshold:
        #         valid_operators.append(op)
        #         valid_operators_qubit.append(op_q)

        # operators = valid_operators
        # self.operators = valid_operators  # Update operators
        # operators_qubit = valid_operators_qubit

        # self.total_circuit_runs = total_circuit_runs

    def run(self, circuit, uniform_shot_allocation=True) -> QSEResult:
        """
        Run the QSE protocol using the given circuit.

        Args:
            circuit: Qibo circuit used to prepare the reference state (e.g. from VQE).
            statevector: Optional exact statevector. If None and n_shots is None,
                         it is automatically computed via circuit().state().

        Returns:
            QSEResult containing the energies and eigenvectors in the subspace,
            plus measurement-cost metadata when sampling is used.
        """
        # Construct H and S matrices
        self.calculate_hs_matrices(circuit, uniform_shot_allocation)

        # Solve generalized eigenvalue problem: H_LR C = S_LR C E
        # Since S_LR can still be ill-conditioned, we diagonalize S_LR first
        s_evals, s_evecs = np.linalg.eigh(self.S)

        # Keep only eigenvectors of S where eigenvalue is > threshold
        valid_idx = s_evals > self.config.eigenvalue_threshold
        if not np.any(valid_idx):
            raise ValueError(
                "No valid QSE subspace found after overlap matrix diagonalization. "
                "The excitation_threshold or eigenvalue_threshold might be too strict, "
                "or the reference state provides no valid excitations."
            )

        s_evals = s_evals[valid_idx]
        s_evecs = s_evecs[:, valid_idx]  # Projection matrix P

        # S^{-1/2}
        S_inv_half = s_evecs @ np.diag(1.0 / np.sqrt(s_evals))

        # Form the orthogonalized Hamiltonian H' = S^{-1/2}^T H S^{-1/2}
        H_prime = S_inv_half.T.conj() @ self.H @ S_inv_half

        eigenvalues, C_prime = np.linalg.eigh(H_prime)

        # Transform back to original basis C = S^{-1/2} C'
        eigenvectors = S_inv_half @ C_prime

        return QSEResult(
            energies=eigenvalues,
            eigenvectors=eigenvectors,
            excitation_operators=self.operators,
            subspace_dimension=self.dim,
            projected_subspace_dimension=len(s_evals),
            # total_circuit_runs=self.total_circuit_runs,
        )


def qse(molecule, circuit, config=None) -> QSEResult:
    """Convenience wrapper for Quantum Subspace Expansion."""
    runner = QSE(molecule, config=config)
    # Runner handles computing the expectation values using either exact statevectors or circuit samples
    return runner.run(circuit)
