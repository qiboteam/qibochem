"""
Functions for optimising the measurement cost of obtaining the expectation value
"""

from math import prod

import numpy as np
from qibo import Circuit, gates, symbols
from qibo.gates import Gate
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z
from sympy.core.expr import Expr
from sympy.core.numbers import One

from qibochem.ansatz.ucc import expi_pauli
from qibochem.measurement.util import (
    _binary_gaussian_elimination,
    _binary_nullspace,
    _get_sigma_terms,
    _group_commuting_terms,
    _lagrangian_subspace,
    _pauli_to_symplectic,
    _phase_factor,
    _solve_linear_system,
    _sort_tau_terms,
    _symplectic_to_pauli,
    _synthesise_circuit,
)


def _term_to_string(term: Expr) -> str:
    """Convert a single Pauli term to its string representation; dropping any coefficient"""
    return " ".join(str(_x) for _x in term.args if isinstance(_x, (X, Y, Z))) if term.args else str(term)


def _u_circuit(tau_terms: list[str], sigma_terms: list[str], nqubits: int) -> Circuit:
    """
    Circuit formulation by Izmaylov and co-workers for measuring generally commuting terms simultaneously.
    TODO: Consider using the gates from Qibo directly, instead of expi_pauli
    """
    circuit = Circuit(nqubits)
    for _tau, _sigma in zip(tau_terms, sigma_terms):
        # Convert the strings to QubitOperators
        tau_i = " ".join(_tau)
        sigma_i = " ".join(_sigma)

        theta = 0.25 * np.pi
        # Build up the circuit
        circuit += expi_pauli(nqubits, sigma_i, theta)
        circuit += expi_pauli(nqubits, tau_i, theta)
        circuit += expi_pauli(nqubits, sigma_i, theta)

    return circuit


def _qwc_measurement_gates(expression: Expr) -> list[Gate]:
    """
    Measurement gates to be added to the circuit for an expression of qubit-wise commuting terms. Resultant measurements
    can be used to calculate the expectation values of ALL terms in expression directly.
    """
    m_gates, _m_gates = {}, {}
    # Single Pauli operator
    if not expression.args:
        return [gates.M(expression.target_qubit, basis=type(expression.gate))]
    # Either a single Pauli term or a sum of Pauli terms
    for term in expression.args:
        # Term should either be a single Pauli operator or a Pauli string
        if isinstance(term, (X, Y, Z)):
            _m_gates = {term.target_qubit: gates.M(term.target_qubit, basis=type(term.gate))}
        else:
            _m_gates = {
                pauli_op.target_qubit: gates.M(pauli_op.target_qubit, basis=type(pauli_op.gate))
                for pauli_op in term.args
                if hasattr(pauli_op, "target_qubit") and m_gates.get(pauli_op.target_qubit) is None
            }
        m_gates = {**m_gates, **_m_gates}
    return sorted(m_gates.values(), key=lambda x: x.target_qubits)


def _qwc_measurements(hamiltonian: SymbolicHamiltonian) -> list[tuple[Expr, list[Gate], list[Gate]]]:
    """
    Sort the Hamiltonian into separate groups of mutually qubitwise commuting terms, and returns the grouped terms
    along with their associated measurement gates
    """
    # Build dictionary with keys = string representation of the terms, values = corresponding (sympy.Expr, term coeff)
    if hamiltonian.form.args:
        ham_terms = {
            _term_to_string(term): (term, coeff)
            for term, coeff in hamiltonian.form.as_coefficients_dict().items()
            if not isinstance(term, One)
        }
    else:
        ham_terms = {_term_to_string(hamiltonian.form): (hamiltonian.form, 1.0)}  # Single Pauli operator
    term_groups = _group_commuting_terms(ham_terms.keys(), qubitwise=True)
    return [
        (
            sum(ham_terms[term][1] * ham_terms[term][0] for term in term_group),  # Original expression: coeff*term
            _qwc_measurement_gates(
                sum(ham_terms[term][0] for term in term_group)
            ),  # No coeff for _qwc_measurement_gates
            [],  # No additional rotation gates needed; Already included in `basis` argument of gates.M
        )
        for term_group in term_groups
    ]


def _gc_measurement_mapping(expression: Expr, nqubits: int, method: str) -> tuple[dict[str, Expr], list[Gate]]:
    """
    Basis rotation gates to be added to the circuit for generally commuting terms. Resultant measurements
    can be used to calculate the expectation values of ALL the terms in expression directly.

    Args:
        expression (sympy.Expr): Group of Pauli terms that all mutually commute with each other qubitwise
        nqubits (int): Number of qubits of the original Hamiltonian
        method (str): Circuit formulation to use, either "chong" (default) or "izmaylov"

    Returns:
        tuple[dict[str, Expr], list[Gate]]: (Mapping of original expression, Gates to add to original Qibo circuit)
    """
    # Single Pauli operator
    if not expression.args:
        return {_term_to_string(expression): expression}, [
            gates.M(expression.target_qubit, basis=type(expression.gate))
        ]
    # Otherwise, expression is a sum of terms
    term_list = [_term_to_string(term) for term in expression.args if _term_to_string(term)[0] in ("X", "Y", "Z")]
    v_subspace = np.array([_pauli_to_symplectic(terms.split(), nqubits) for terms in term_list])
    v_basis = _binary_gaussian_elimination(v_subspace)

    dim_v = v_basis.shape[0]
    dim_symplectic = v_basis.shape[1] // 2
    # If dim(V) < N, update v_basis to form a Lagrangian subspace
    if dim_v != dim_symplectic:
        nullspace = _binary_nullspace(v_basis)
        # Interchange the 1st/2nd half of the indices to get nullspace in a symplectic sense
        nullspace = np.concatenate((nullspace[:, dim_symplectic:], nullspace[:, :dim_symplectic]), axis=1)
        v_basis = _lagrangian_subspace(nullspace)
    # Different methods of circuit synthesis
    if method == "chong":
        x_result = _solve_linear_system(v_basis, v_subspace)
        phase_factors = [_phase_factor(v_basis[pauli_op]) for pauli_op in x_result]
        u_gates = _synthesise_circuit(v_basis)
        mapping = {
            term: phase * prod(Z(_i) for _i in soln) for term, phase, soln in zip(term_list, phase_factors, x_result)
        }
    elif method == "izmaylov":
        v_basis = _sort_tau_terms(v_basis)
        new_tau_terms, sigma_terms = _get_sigma_terms(v_basis)
        x_result = _solve_linear_system(new_tau_terms, v_subspace)
        phase_factors = [_phase_factor(new_tau_terms[pauli_op]) for pauli_op in x_result]
        tau_term_str = [_symplectic_to_pauli(tau_i) for tau_i in new_tau_terms]
        sigma_term_str = [_symplectic_to_pauli(sigma_i) for sigma_i in sigma_terms]
        qwc_terms = [_symplectic_to_pauli(sum(sigma_terms[_x] for _x in pauli_op)) for pauli_op in x_result]
        mapping = {
            term: phase * prod([getattr(symbols, sigma[0])(int(sigma[1:])) for sigma in pauli_op])
            for term, phase, pauli_op in zip(term_list, phase_factors, qwc_terms)
        }
        # Define the measurement gates
        u_gates = _u_circuit(tau_term_str, sigma_term_str, nqubits).queue
    else:
        raise ValueError("Unknown method!")
    return mapping, u_gates


def _gc_measurements(hamiltonian: SymbolicHamiltonian, method: str) -> list[tuple[Expr, list[Gate], list[Gate]]]:
    """
    Sort the Hamiltonian terms into separate groups of mutually commuting terms, and returns the updated expressions to
    measured, their associated measurement gates, and the rotation gates to update the initial expressions
    """
    # Build dictionary with keys = string representation of the terms, values = corresponding (sympy.Expr, term coeff)
    if hamiltonian.form.args:
        ham_terms = {
            _term_to_string(term): (term, coeff)
            for term, coeff in hamiltonian.form.as_coefficients_dict().items()
            if not isinstance(term, One)
        }
    else:
        ham_terms = {_term_to_string(hamiltonian.form): (hamiltonian.form, 1.0)}  # Single Pauli operator
    term_groups = _group_commuting_terms(ham_terms.keys(), qubitwise=False)
    to_return = []
    for term_group in term_groups:
        # Check for qubitwise commutativity
        qubitwise_commutative = len(_group_commuting_terms(term_group, qubitwise=True)) == 1
        if qubitwise_commutative:
            new_expression = sum(ham_terms[term][1] * ham_terms[term][0] for term in term_group)  # Unchanged if QWC
            rotation_gates = []
        else:
            grouped_expression = sum(ham_terms[term][0] for term in term_group)
            mapping, rotation_gates = _gc_measurement_mapping(grouped_expression, hamiltonian.nqubits, method)
            # Update the initial expression based on the obtained mapping
            new_expression = sum(ham_terms[term][1] * mapping[term] for term in term_group)
        # Add measurement gates based on the updated expression
        measurement_gates = _qwc_measurement_gates(new_expression)
        to_return.append((new_expression, measurement_gates, rotation_gates))
    return to_return


def _measurement_basis_rotations(
    hamiltonian: SymbolicHamiltonian, grouping: str | None = None
) -> list[tuple[Expr, list[Gate], list[Gate]]]:
    """
    Sort Hamiltonian into separate groups and get the basis rotation gates to be applied for each of the corresponding
    (group of) terms in the Hamiltonian. `grouping` argument must be in (None, "qwc", "gc", "gc2")
    """
    result = []
    if grouping is None:
        result += [
            (coeff * term, _qwc_measurement_gates(term), [])
            for term, coeff in hamiltonian.form.as_coefficients_dict().items()
            if not isinstance(term, One)  # Ignore any constant term
        ]
    elif grouping == "qwc":
        result += _qwc_measurements(hamiltonian)
    elif grouping == "gc":
        result += _gc_measurements(hamiltonian, "chong")
    elif grouping == "gc2":
        result += _gc_measurements(hamiltonian, "izmaylov")
    else:
        raise NotImplementedError("Unknown Pauli term grouping method!")
    return result
