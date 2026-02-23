"""Quantum Subspace Expansion (QSE)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openfermion


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
    total_circuit_runs: int = 0


class QSE:
    def __init__(self, molecule, config=None):
        """
        Quantum Subspace Expansion (QSE) manager.

        Args:
            molecule: `qibochem.driver.Molecule` instance.
            config: Configuration for QSE.
        """
        self.molecule = molecule
        self.config = config or QSEConfig()

    def run(self, circuit, statevector=None) -> QSEResult:
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
        if statevector is None and self.config.n_shots is None:
            statevector = circuit().state()

        total_circuit_runs = 0

        # Determine number of active spin-orbitals (use active space if specified)
        n_active_orbs = self.molecule.n_active_orbs if self.molecule.n_active_orbs is not None else self.molecule.nso

        # Generate excitation operators
        operators = self._generate_excitation_operators(n_active_orbs)
        operators_qubit = [self._fermion_to_qubit(op) for op in operators]

        # Collect S_aa terms for filtering
        S_aa_qs = []
        pauli_strings_Saa = set()
        for op_q in operators_qubit:
            op_q_dag = openfermion.hermitian_conjugated(op_q)
            S_aa_q = op_q_dag * op_q
            S_aa_qs.append(S_aa_q)
            if self.config.n_shots is not None:
                pauli_strings_Saa.update(S_aa_q.terms.keys())

        if self.config.n_shots is not None:
            Saa_exp_vals, saa_runs = self._measure_pauli_strings(
                pauli_strings_Saa,
                circuit,
                self.config.n_shots,
                return_metadata=True,
            )
            total_circuit_runs += saa_runs

        # Filter out operators that annihilate the state to avoid singular overlap matrix
        valid_operators = []
        valid_operators_qubit = []
        for op, op_q, S_aa_q in zip(operators, operators_qubit, S_aa_qs):
            if self.config.n_shots is not None:
                norm_sq = np.real(sum(coeff * Saa_exp_vals[pauli] for pauli, coeff in S_aa_q.terms.items()))
            else:
                norm_sq = np.real(self._expectation_value_qubit(S_aa_q, statevector, n_active_orbs))

            if norm_sq > self.config.excitation_threshold:
                valid_operators.append(op)
                valid_operators_qubit.append(op_q)

        operators = valid_operators
        operators_qubit = valid_operators_qubit

        dim = len(operators)
        H_LR = np.zeros((dim, dim), dtype=complex)
        S_LR = np.zeros((dim, dim), dtype=complex)

        ferm_hamiltonian = self.molecule.hamiltonian(ham_type="f", ferm_qubit_map=self.config.ferm_qubit_map)
        H_qubit = self._fermion_to_qubit(ferm_hamiltonian)

        S_ab_qs = {}
        H_ab_qs = {}
        pauli_strings_main = set()

        for a, E_a_q in enumerate(operators_qubit):
            E_a_q_dag = openfermion.hermitian_conjugated(E_a_q)
            for b, E_b_q in enumerate(operators_qubit):
                if a > b:
                    continue

                S_ab_q = E_a_q_dag * E_b_q
                H_ab_q = E_a_q_dag * H_qubit * E_b_q

                S_ab_qs[(a, b)] = S_ab_q
                H_ab_qs[(a, b)] = H_ab_q

                if self.config.n_shots is not None:
                    pauli_strings_main.update(S_ab_q.terms.keys())
                    pauli_strings_main.update(H_ab_q.terms.keys())

        if self.config.n_shots is not None:
            main_exp_vals, main_runs = self._measure_pauli_strings(
                pauli_strings_main,
                circuit,
                self.config.n_shots,
                return_metadata=True,
            )
            total_circuit_runs += main_runs

        for a in range(dim):
            for b in range(dim):
                if a > b:
                    continue

                if self.config.n_shots is not None:
                    S_ab_val = sum(coeff * main_exp_vals[pauli] for pauli, coeff in S_ab_qs[(a, b)].terms.items())
                    H_ab_val = sum(coeff * main_exp_vals[pauli] for pauli, coeff in H_ab_qs[(a, b)].terms.items())
                else:
                    S_ab_val = self._expectation_value_qubit(S_ab_qs[(a, b)], statevector, n_active_orbs)
                    H_ab_val = self._expectation_value_qubit(H_ab_qs[(a, b)], statevector, n_active_orbs)

                S_LR[a, b] = S_ab_val
                H_LR[a, b] = H_ab_val

                if a != b:
                    S_LR[b, a] = np.conj(S_ab_val)
                    H_LR[b, a] = np.conj(H_ab_val)

        # Solve generalized eigenvalue problem: H_LR C = S_LR C E
        # Since S_LR can still be ill-conditioned, we diagonalize S_LR first
        s_evals, s_evecs = np.linalg.eigh(S_LR)

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
        H_prime = S_inv_half.T.conj() @ H_LR @ S_inv_half

        eigenvalues, C_prime = np.linalg.eigh(H_prime)

        # Transform back to original basis C = S^{-1/2} C'
        eigenvectors = S_inv_half @ C_prime

        return QSEResult(
            energies=eigenvalues,
            eigenvectors=eigenvectors,
            excitation_operators=operators,
            subspace_dimension=dim,
            projected_subspace_dimension=len(s_evals),
            total_circuit_runs=total_circuit_runs,
        )

    def _fermion_to_qubit(self, ferm_op: openfermion.FermionOperator) -> openfermion.QubitOperator:
        """Map FermionOperator to QubitOperator according to config."""
        if self.config.ferm_qubit_map == "jw":
            qubit_op = openfermion.jordan_wigner(ferm_op)
        elif self.config.ferm_qubit_map == "bk":
            qubit_op = openfermion.bravyi_kitaev(ferm_op)
        else:
            raise KeyError(f"Unknown fermion->qubit mapping: {self.config.ferm_qubit_map}")
        qubit_op.compress()
        return qubit_op

    def _measure_pauli_strings(self, pauli_strings: set, circuit, n_shots: int, return_metadata: bool = False):
        """Measure expectation values of Pauli strings and optionally return measurement metadata."""
        from functools import reduce

        from qibo.symbols import X, Y, Z

        from qibochem.measurement.optimization import (
            group_commuting_terms,
            qwc_measurement_gates,
        )
        from qibochem.measurement.result import pauli_term_measurement_expectation

        pauli_qibo = []
        pauli_mapping = {}
        for term in pauli_strings:
            if not term:
                continue
            qibo_str = " ".join(f"{op}{idx}" for idx, op in term)
            pauli_qibo.append(qibo_str)
            pauli_mapping[term] = qibo_str

        term_groups = group_commuting_terms(pauli_qibo, qubitwise=True)
        exp_vals = {}
        circuit_runs = 0

        for term_group in term_groups:
            group_exprs = []
            for q_str in term_group:
                factors = []
                for factor_str in q_str.split():
                    op = factor_str[0]
                    idx = int(factor_str[1:])
                    if op == "X":
                        factors.append(X(idx))
                    elif op == "Y":
                        factors.append(Y(idx))
                    elif op == "Z":
                        factors.append(Z(idx))
                expr = reduce(lambda x, y: x * y, factors, 1) if factors else 1
                group_exprs.append((q_str, expr))

            circuit_runs += 1

            sum_expr = sum(expr for _, expr in group_exprs)
            meas_gates = qwc_measurement_gates(sum_expr)

            _circuit = circuit.copy()
            _circuit.add(meas_gates)
            result = _circuit(nshots=n_shots)
            freqs = result.frequencies(binary=True)
            qubit_map = sorted(qubit for gate in meas_gates for qubit in gate.target_qubits)

            for q_str, expr in group_exprs:
                if not freqs:
                    val = 0.0
                else:
                    val = pauli_term_measurement_expectation(expr, freqs, qubit_map)
                exp_vals[q_str] = val

        final_dict = {(): 1.0}
        for term in pauli_strings:
            if term:
                final_dict[term] = exp_vals[pauli_mapping[term]]
        if return_metadata:
            return final_dict, circuit_runs
        return final_dict

    def _expectation_value_qubit(self, qubit_op: openfermion.QubitOperator, statevector, n_qubits: int) -> complex:
        """Helper function to calculate the expectation value of a QubitOperator on a statevector."""
        if not qubit_op.terms:
            return 0.0

        keys = list(qubit_op.terms.keys())
        if keys == [()]:
            return qubit_op.terms[()]

        sparse_mat = openfermion.get_sparse_operator(qubit_op, n_qubits=n_qubits)
        return np.vdot(statevector, sparse_mat.dot(statevector))

    def _generate_excitation_operators(
        self, nso: int, to_qubit_op: bool = False
    ) -> list[openfermion.FermionOperator | openfermion.QubitOperator]:
        """Generate one-body excitation operators + the identity."""
        operators = [openfermion.FermionOperator("")]
        for i in range(nso):
            for j in range(nso):
                if self.config.conserve_spin and (i % 2) != (j % 2):
                    continue
                operators.append(openfermion.FermionOperator(f"{i}^ {j}"))
        if to_qubit_op:
            operators = [self._fermion_to_qubit(op) for op in operators]
        return operators


def qse(molecule, circuit, config=None) -> QSEResult:
    """Convenience wrapper for Quantum Subspace Expansion."""
    runner = QSE(molecule, config=config)
    # Runner handles computing the expectation values using either exact statevectors or circuit samples
    return runner.run(circuit)
