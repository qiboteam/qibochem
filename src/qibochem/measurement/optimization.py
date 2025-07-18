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


def check_terms_commutativity(term1: str, term2: str, qubitwise: bool):
    """
    Check if terms 1 and 2 are mutually commuting. The 'qubitwise' flag determines if the check is for general
    commutativity (False), or the stricter qubitwise commutativity.

    Args:
        term1/term2: Strings representing a single Pauli term. E.g. "X0 Z1 Y3". Obtained from a Qibo SymbolicTerm as
        ``" ".join(factor.name for factor in term.factors)``.
        qubitwise (bool): Determines if the check is for general commutativity, or the stricter qubitwise commutativity

    Returns:
        bool: Do terms 1 and 2 commute?
    """
    # Get a list of common qubits for each term
    common_qubits = {_term[1:] for _term in term1.split() if _term[0] != "I"} & {
        _term[1:] for _term in term2.split() if _term[0] != "I"
    }
    if not common_qubits:
        return True
    # Get the single Pauli operators for the common qubits for both Pauli terms
    term1_ops = [_op for _op in term1.split() if _op[1:] in common_qubits]
    term2_ops = [_op for _op in term2.split() if _op[1:] in common_qubits]
    if qubitwise:
        # Qubitwise: Compare the Pauli terms at the common qubits. Any difference => False
        return all(_op1 == _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # General commutativity:
    # Get the number of single Pauli operators that do NOT commute
    n_noncommuting_ops = sum(_op1 != _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # term1 and term2 have general commutativity iff n_noncommuting_ops is even
    return n_noncommuting_ops % 2 == 0


def group_commuting_terms(terms_list, qubitwise):
    """
    Groups the terms in terms_list into as few groups as possible, where all the terms in each group commute
    mutually == Finding the minimum clique cover (i.e. as few cliques as possible) for the graph whereby each node
    is a Pauli string, and an edge exists between two nodes iff they commute.

    This is equivalent to the graph colouring problem of the complement graph (i.e. edge between nodes if they DO NOT
    commute), which this function follows.

    Args:
        terms_list: List of strings. The strings should follow the output from
            ``" ".join(factor.name for factor in term.factors)``, where term is a Qibo SymbolicTerm. E.g. "X0 Z1".
        qubitwise: Determines if the check is for general commutativity, or the stricter qubitwise commutativity

    Returns:
        list: Containing groups (lists) of Pauli strings that all commute mutually
    """
    G = nx.Graph()
    # Complement graph: Add all the terms as nodes first, then add edges between nodes if they DO NOT commute
    G.add_nodes_from(terms_list)
    G.add_edges_from(
        (term1, term2)
        for _i1, term1 in enumerate(terms_list)
        for _i2, term2 in enumerate(terms_list)
        if _i2 > _i1 and not check_terms_commutativity(term1, term2, qubitwise)
    )
    # Solve using Greedy Colouring on NetworkX
    sorted_groups = nx.coloring.greedy_color(G)
    group_ids = set(sorted_groups.values())
    # Sort results so that test results will be replicable
    term_groups = sorted(
        sorted(group for group, group_id in sorted_groups.items() if group_id == _id) for _id in group_ids
    )
    return term_groups


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
    else:
        raise NotImplementedError("Not ready yet!")
    return result
