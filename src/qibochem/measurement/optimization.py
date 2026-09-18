"""
Functions for optimising the measurement cost of obtaining the expectation value
"""

import numpy as np
from qibo import Circuit, gates
from qibo.gates import Gate
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z
from sympy import Add, Mul
from sympy.core.expr import Expr
from sympy.core.numbers import One

from qibochem.measurement.util import (
    _binary_gaussian_elimination,
    _binary_nullspace,
    _get_sigma_terms,
    _group_commuting_terms,
    _lagrangian_subspace,
    _phase_factor,
    _solve_linear_system,
    _sort_tau_terms,
    _symplectic_to_pauli,
    _synthesise_circuit,
)


def _u_circuit(tau_terms: Expr, sigma_terms: Expr, nqubits: int) -> Circuit:
    """
    Construct the rotation circuit using a SymbolicHamiltonian.circuit
    """
    circuit = Circuit(nqubits)
    theta = -0.25 * np.pi
    for tau_term, sigma_term in zip(tau_terms, sigma_terms):
        tau_hamiltonian = SymbolicHamiltonian(tau_term, nqubits=nqubits)
        sigma_hamiltonian = SymbolicHamiltonian(sigma_term, nqubits=nqubits)
        circuit += sigma_hamiltonian.circuit(theta)
        circuit += tau_hamiltonian.circuit(theta)
        circuit += sigma_hamiltonian.circuit(theta)
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


def _gc_measurement_mapping(
    term_group: list[Expr], term_dict: dict[Expr, tuple[float, np.ndarray]], method: str
) -> tuple[dict[str, Expr], list[Gate]]:
    """
    Basis rotation gates to be added to the circuit for generally commuting terms. Resultant measurements
    can be used to calculate the expectation values of ALL the terms in expression directly.

    Args:
        term_group (list[sympy.Expr]): Group of Pauli terms that mutually commutes with each other
        term_dict (dict[Expr, tuple[float, np.ndarray]]): Coefficients and symplectic form of the Pauli terms in the
            Hamiltonian
        method (str): Circuit formulation to use, either "chong" (default) or "izmaylov"

    Returns:
        tuple[dict[Expr, Expr], list[Gate]]: (Mapping of original expression, Gates to add to original Qibo circuit)
    """
    v_subspace = np.array([term_dict[term][1] for term in term_group], dtype=np.uint8)
    v_basis = _binary_gaussian_elimination(v_subspace)

    dim_v = v_basis.shape[0]
    dim_symplectic = v_basis.shape[1] // 2
    # If dim(V) < N, update v_basis to form a Lagrangian subspace
    if dim_v != dim_symplectic:
        nullspace = _binary_nullspace(v_basis)
        # Interchange the 1st/2nd half of the indices to get nullspace in a symplectic sense
        nullspace = nullspace[:, np.r_[dim_symplectic : 2 * dim_symplectic, 0:dim_symplectic]]
        nullspace = _binary_gaussian_elimination(nullspace)
        v_basis = _lagrangian_subspace(nullspace)

    # Different methods of circuit synthesis
    if method == "chong":
        x_result = _solve_linear_system(v_basis, v_subspace)
        phase_factors = [_phase_factor(v_basis[pauli_op]) for pauli_op in x_result]
        u_gates, phases = _synthesise_circuit(v_basis)
        mapping = {
            term: Mul(phase, *(phases[i] * Z(i) for i in soln))
            for term, phase, soln in zip(term_group, phase_factors, x_result)
        }
    elif method == "izmaylov":
        v_basis = _sort_tau_terms(v_basis)
        tau_terms, sigma_terms = _get_sigma_terms(v_basis)
        x_result = _solve_linear_system(tau_terms, v_subspace)
        phase_factors = [_phase_factor(tau_terms[pauli_op]) for pauli_op in x_result]
        # Convert tau/sigma_terms from np.array -> sympy.Expr
        tau_terms = [_symplectic_to_pauli(tau_i) for tau_i in tau_terms]
        sigma_terms = [_symplectic_to_pauli(sigma_i) for sigma_i in sigma_terms]
        qwc_terms = [Mul(*(sigma_terms[_x] for _x in pauli_op)) for pauli_op in x_result]
        mapping = {term: Mul(phase, qwc_term) for term, phase, qwc_term in zip(term_group, phase_factors, qwc_terms)}
        # Define the measurement gates
        u_gates = _u_circuit(tau_terms, sigma_terms, v_basis.shape[1]).queue  # Extract nqubits from vector
    else:
        raise ValueError("Unknown method!")
    return mapping, u_gates


def _gc_measurements(
    term_dict: dict[Expr, tuple[float, np.ndarray]], term_groups: list[list[Expr]], method: str
) -> list[tuple[Expr, list[Gate], list[Gate]]]:
    """
    Sort the Hamiltonian terms into separate groups of mutually commuting terms, and returns the updated expressions to
    be measured, their associated measurement gates, and the rotation gates to update the initial expressions

    Args:
        term_dict (dict[Expr, tuple[float, np.ndarray]]): Coefficients and symplectic form of the Pauli terms in the
            Hamiltonian
        term_groups (list[list[Expr]]): Grouped terms
        method (str): How the rotation circuit is constructed; must be in ("gc", "gc2")

    Returns:
        list[tuple[Expr, list[Gate], list[Gate]]]: Updated expressions to be measured, their associated measurement
            gates, and the rotation gates to update the initial expressions
    """
    result = []
    for term_group in term_groups:
        mapping, rotation_gates = _gc_measurement_mapping(term_group, term_dict, method)
        # Update the initial expression based on the obtained mapping
        new_expression = Add(*(term_dict[term][0] * mapping[term] for term in term_group), evaluate=False)
        # Add measurement gates based on the updated expression
        measurement_gates = _qwc_measurement_gates(new_expression)
        result.append((new_expression, measurement_gates, rotation_gates))
    return result


def _measurement_basis_rotations(
    hamiltonian: SymbolicHamiltonian, grouping: str | None, method: str
) -> list[tuple[Expr, list[Gate], list[Gate]]]:
    """
    Sort Hamiltonian into separate groups and get the basis rotation gates to be applied for each of the corresponding
    (group of) terms in the Hamiltonian. `grouping` argument

    Args:
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Molecular Hamiltonian
        grouping (str | None): How to group and construct the rotation circuit; Must be in (None, "qwc", "gc", "gc2")
        method (str): Algorithm used to group compatible Pauli terms. Must be in ("sorted", "greedy", "graph")

    Returns:
        tuple[list[tuple[Expr, list[Gate], list[Gate]]], float]:
            Grouped terms along with the associated measurement and rotation gates respectively, and the constant term
            in the Hamiltonian
    """
    result, constant = [], 0.0
    if grouping is None:
        for term, coeff in hamiltonian.form.as_coefficients_dict().items():
            if not isinstance(term, One):
                result.append((coeff * term, _qwc_measurement_gates(term), []))
            else:
                constant += coeff
        return result, constant

    # Grouping of Pauli terms
    term_dict, term_groups = _group_commuting_terms(hamiltonian, grouping == "qwc", method)
    # Extract the constant term from term_dict
    constant += term_dict.get(1, [0.0])[0]

    if grouping == "qwc":
        result = [
            (
                # Original expression: coeff*term
                Add(*(term_dict[term][0] * term for term in term_group), evaluate=False),
                _qwc_measurement_gates(Add(*term_group, evaluate=False)),
                [],  # No additional rotation gates needed; Already included in `basis` argument of gates.M
            )
            for term_group in term_groups
        ]
    elif grouping == "gc":
        result = _gc_measurements(term_dict, term_groups, "chong")
    elif grouping == "gc2":
        result = _gc_measurements(term_dict, term_groups, "izmaylov")
    else:
        raise NotImplementedError("Unknown Pauli term grouping method!")
    return result, constant
