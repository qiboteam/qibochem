"""
Functions for optimising the measurement cost of obtaining the expectation value
"""

import networkx as nx
from qibo import gates
from qibo.symbols import X, Y, Z
from sympy.core.numbers import One


def term_to_string(term):
    """
    Convert a single Pauli term (:class:`sympy.Expr`) to its string representation. Drops the coefficient and will not
    check if input is a float!!
    """
    return " ".join(str(_x) for _x in term.args if isinstance(_x, (X, Y, Z))) if term.args else str(term)


from qibochem.measurement.util import check_terms_commutativity, group_commuting_terms


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


def gc_measurement_gates(expression):
    """
    Get the list of (basis rotation) measurement gates to be added to the circuit. The measurements from the resultant
    circuit can then be used to obtain the expectation values of ALL the terms in expression directly.

    Args:
        expression (sympy.Expr): Group of Pauli terms that all mutually commute with each other qubitwise

    Returns:
        list: Measurement gates to be appended to the Qibo circuit
    """

    # m_gates, _m_gates = {}, {}
    # # Single Pauli operator
    # if not expression.args:
    #     return [gates.M(expression.target_qubit, basis=type(expression.gate))]
    # # Either a single Pauli term or a sum of Pauli terms
    # for term in expression.args:
    #     # Term should either be a single Pauli operator or a Pauli string
    #     if isinstance(term, (X, Y, Z)):
    #         _m_gates = {term.target_qubit: gates.M(term.target_qubit, basis=type(term.gate))}
    #     else:
    #         _m_gates = {
    #             pauli_op.target_qubit: gates.M(pauli_op.target_qubit, basis=type(pauli_op.gate))
    #             for pauli_op in term.args
    #             if m_gates.get(pauli_op.target_qubit) is None
    #         }
    #     m_gates = {**m_gates, **_m_gates}
    return  # list(m_gates.values())


def grouped_measurements(hamiltonian, generally_commuting=False):
    """
    Sort out a list of Hamiltonian terms into separate groups of mutually qubitwise commuting terms, and returns the
    grouped terms along with their associated measurement gates

    Args:
        hamiltonian (:class:`qibo.hamiltonians.SymbolicHamiltonian`): Hamiltonian of interest
        generally_commuting (bool): "GC" or "QWC" measurements. Default (False) is "QWC"

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
    term_groups = group_commuting_terms(ham_terms.keys(), qubitwise=(not generally_commuting))
    measurement_gates_fn = qwc_measurement_gates if not generally_commuting else gc_measurement_gates
    return [
        (
            sum(ham_terms[term][1] * ham_terms[term][0] for term in term_group),  # Original expression: coeff*term
            measurement_gates_fn(sum(ham_terms[term][0] for term in term_group)),  # No coeff for qwc_measurement_gates
        )
        for term_group in term_groups
    ]


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
        result += grouped_measurements(hamiltonian, False)
    elif grouping == "gc":
        result += grouped_measurements(hamiltonian, True)
    else:
        raise NotImplementedError("Not ready yet!")
    return result
