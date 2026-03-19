"""
Functions for optimising the measurement cost of obtaining the expectation value
"""

from math import prod

import numpy as np
from qibo import Circuit, gates, symbols
from qibo.symbols import X, Y, Z
from sympy.core.numbers import One

from qibochem.ansatz.ucc import expi_pauli
from qibochem.measurement.util import (
    binary_gaussian_elimination,
    binary_nullspace,
    get_sigma_terms,
    group_commuting_terms,
    langrangian_subspace,
    pauli_to_symplectic,
    phase_factor,
    solve_linear_system,
    sort_tau_terms,
    symplectic_to_pauli,
)


def term_to_string(term):
    """
    Convert a single Pauli term (:class:`sympy.Expr`) to its string representation. Drops the coefficient and will not
    check if input is a float!!
    """
    return " ".join(str(_x) for _x in term.args if isinstance(_x, (X, Y, Z))) if term.args else str(term)


def u_circuit(tau_terms, sigma_terms, n_qubits):
    """
    Obtain the ciruit representing the unitary transformation to be applied on top of the original circuit ansatz
    to allow the expectation value of a group of Pauli terms (that commute) to be obtained using qubit-wise commuting
    measurements
    TODO: Make nice the docstring

    Args:
        tau_terms, sigma_terms: Lists of strings representing the decomposition of the group of Pauli terms

    Returns:
        Qibo Circuit
    """
    circuit = Circuit(n_qubits)
    for _tau, _sigma in zip(tau_terms, sigma_terms):
        # Convert the strings to QubitOperators
        tau_i = " ".join(_tau)
        sigma_i = " ".join(_sigma)

        theta = 0.25 * np.pi
        # Build up the circuit
        circuit += expi_pauli(n_qubits, sigma_i, theta)
        circuit += expi_pauli(n_qubits, tau_i, theta)
        circuit += expi_pauli(n_qubits, sigma_i, theta)

    return circuit


def qwc_measurement_gates(expression):
    """
    Get the list of (basis rotation) measurement gates to be added to the circuit. The measurements from the resultant
    circuit can then be used to obtain the expectation values of ALL the terms in expression directly.

    Args:
        expression (sympy.Expr): Group of Pauli terms that all mutually commute with each other qubitwise

    Returns:
        list: Measurement gates to be appended to the Qibo circuit
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
                if m_gates.get(pauli_op.target_qubit) is None
            }
        m_gates = {**m_gates, **_m_gates}
    return list(m_gates.values())


def qwc_measurements(hamiltonian):
    """
    Sort out a list of Hamiltonian terms into separate groups of mutually qubitwise commuting terms, and returns the
    grouped terms along with their associated measurement gates

    Args:
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Hamiltonian of interest

    Returns:
        list: List of two-tuples, with each tuple given as (sorted_ham, [`list of measurement gates`]), where
            sorted_ham is a :class:`qibo.hamiltonians.SymbolicHamiltonian`
    """
    # Build dictionary with keys = string representation of the terms, values = corresponding (sympy.Expr, term coeff)
    if hamiltonian.form.args:
        ham_terms = {
            term_to_string(term): (term, coeff)
            for term, coeff in hamiltonian.form.as_coefficients_dict().items()
            if not isinstance(term, One)
        }
    else:
        ham_terms = {term_to_string(hamiltonian.form): (hamiltonian.form, 1.0)}  # Single Pauli operator
    term_groups = group_commuting_terms(ham_terms.keys(), qubitwise=True)
    return [
        (
            sum(ham_terms[term][1] * ham_terms[term][0] for term in term_group),  # Original expression: coeff*term
            qwc_measurement_gates(sum(ham_terms[term][0] for term in term_group)),  # No coeff for qwc_measurement_gates
        )
        for term_group in term_groups
    ]


def gc_measurement_mapping(expression, nqubits):
    """
    TODO: Docstring

    Get the list of (basis rotation) measurement gates to be added to the circuit. The measurements from the resultant
    circuit can then be used to obtain the expectation values of ALL the terms in expression directly.

    Args:
        expression (sympy.Expr): Group of Pauli terms that all mutually commute with each other qubitwise

    Returns:
        dict: Mapping of the original Hamiltonian terms to the new Hamiltonian
        list: Measurement gates to be appended to the original Qibo circuit
    """
    # Single Pauli operator
    if not expression.args:
        return {term_to_string(expression): expression}, [gates.M(expression.target_qubit, basis=type(expression.gate))]
    # Work on the entire expression
    term_list = [term_to_string(term) for term in expression.args]
    v_subspace = np.array([pauli_to_symplectic(terms.split(), nqubits) for terms in term_list])
    v_basis = binary_gaussian_elimination(v_subspace)

    dim_v = v_basis.shape[0]
    dim_symplectic = v_basis.shape[1] // 2
    # If dim(V) < N, update v_basis to form a Lagrangian subspace
    if dim_v != dim_symplectic:
        nullspace = binary_nullspace(v_basis)
        # Interchange the 1st/2nd half of the indices to get nullspace in a symplectic sense
        nullspace = np.concatenate((nullspace[:, dim_symplectic:], nullspace[:, :dim_symplectic]), axis=1)
        v_basis = langrangian_subspace(nullspace)
    # ZC NOTE: I have completely forgotten what is all this about...
    v_basis = sort_tau_terms(v_basis)
    new_tau_terms, sigma_terms = get_sigma_terms(v_basis)
    x_result = solve_linear_system(new_tau_terms, v_subspace)
    tau_term_str = [symplectic_to_pauli(tau_i) for tau_i in new_tau_terms]
    sigma_term_str = [symplectic_to_pauli(sigma_i) for sigma_i in sigma_terms]
    qwc_terms = [symplectic_to_pauli(sum(sigma_terms[_x] for _x in pauli_op)) for pauli_op in x_result]
    phase_factors = [phase_factor(new_tau_terms[pauli_op]) for pauli_op in x_result]
    mapping = {
        term: phase * prod([getattr(symbols, sigma[0])(int(sigma[1:])) for sigma in pauli_op])
        for term, phase, pauli_op in zip(term_list, phase_factors, qwc_terms)
    }
    # Define the measurement gates
    u_gates = u_circuit(tau_term_str, sigma_term_str, nqubits).queue
    m_gates = [gates.M(_q) for _q in {gate.target_qubit for gate in u_gates if hasattr(gate, "target_qubit")}]
    return mapping, u_gates + m_gates


def gc_measurements(hamiltonian):
    """
    Sort out a list of Hamiltonian terms into separate groups of mutually qubitwise commuting terms, and returns the
    grouped terms along with their associated measurement gates

    Args:
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Hamiltonian of interest

    Returns:
        list: List of two-tuples, with each tuple given as (sorted_ham, [`list of measurement gates`]), where
            sorted_ham is a :class:`qibo.hamiltonians.SymbolicHamiltonian`
    """
    # Build dictionary with keys = string representation of the terms, values = corresponding (sympy.Expr, term coeff)
    if hamiltonian.form.args:
        ham_terms = {
            term_to_string(term): (term, coeff)
            for term, coeff in hamiltonian.form.as_coefficients_dict().items()
            if not isinstance(term, One)
        }
    else:
        ham_terms = {term_to_string(hamiltonian.form): (hamiltonian.form, 1.0)}  # Single Pauli operator
    term_groups = group_commuting_terms(ham_terms.keys(), qubitwise=False)
    to_return = []
    for term_group in term_groups:
        grouped_expression = sum(ham_terms[term][1] * ham_terms[term][0] for term in term_group)
        print(f"{grouped_expression = }")
        mapping, gates_to_add = gc_measurement_mapping(grouped_expression, hamiltonian.nqubits)
        print(f"{mapping = }")
        # Update the initial expression based on the obtained mapping
        # new_expression = sum(ham_terms[term][1] * mapping[term] for term in term_group)
        new_expression = 0.0
        for term in term_group:
            print(f"{term = }")
            new_expression += ham_terms[term][1] * mapping[term]
        to_return.append((new_expression, gates_to_add))
    return to_return


def measurement_basis_rotations(hamiltonian, grouping=None):
    """
    Split up and sort the Hamiltonian terms to get the basis rotation gates to be applied to a quantum circuit for the
    respective (group of) terms in the Hamiltonian

    Args:
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Hamiltonian of interest
        grouping (str): Whether or not to group Hamiltonian terms together, i.e. use the same set of measurements to get
            the expectation values of a group of terms simultaneously. Default value of ``None`` will not group any
            terms together, while ``"qwc"`` will group qubitwise commuting terms together, and return the measurement
            gates associated with each group of terms

    Returns:
        list: List of two-tuples; the first item in the tuple is a group of Pauli terms (:class:`sympy.Expr`), and the
        second is a list of measurement gates (:class:`qibo.gates.M`) that can be used to get the expectation value
        for the corresponding expression.
    """
    result = []
    if grouping is None:
        result += [
            (coeff * term, qwc_measurement_gates(term))
            for term, coeff in hamiltonian.form.as_coefficients_dict().items()
            if not isinstance(term, One)  # Ignore any constant term
        ]
    elif grouping == "qwc":
        result += qwc_measurements(hamiltonian)
    elif grouping == "gc":
        result += gc_measurements(hamiltonian)
    else:
        raise NotImplementedError("Not ready yet!")
    return result
