"""
Functions for optimising the measurement cost of obtaining the expectation value
"""

import networkx as nx
import numpy as np
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


def allocate_shots(grouped_terms, n_shots, method=None, max_shots_per_term=None):
    """
    Allocate shots to each group of terms in the Hamiltonian for calculating the expectation value of the Hamiltonian.

    Args:
        grouped_terms (list): Output of measurement_basis_rotations; list of two-tuples with the first term a
            :class:`sympy.Expr` and the second the list of corresponding measurement gates (not used here).
        n_shots (int): Total number of shots to be allocated
        method (str): How to allocate the shots. The available options are: ``"c"``/``"coefficients"``: ``n_shots`` is
            distributed based on the relative magnitudes of the term coefficients, ``"u"``/``"uniform"``: ``n_shots``
            is distributed evenly amongst each term. Default value: ``"c"``.
        max_shots_per_term (int): Upper limit for the number of shots allocated to an individual group of terms. If not
            given, will be defined as a fraction (largest coefficient over the sum of all coefficients in the
            Hamiltonian) of ``n_shots``.

    Returns:
        list: A list containing the number of shots to be used for each group of Pauli terms respectively.
    """
    if method is None:
        method = "c"
    if max_shots_per_term is None:
        # Define based on the fraction of the term group with the largest coefficients w.r.t. sum of all coefficients.
        term_coefficients = np.array(
            [
                sum(abs(coeff) for _t, coeff in expression.as_coefficients_dict().items())
                for (expression, _) in grouped_terms
            ],
            dtype=float,
        )
        max_shots_per_term = int(np.ceil(n_shots * (np.max(term_coefficients) / sum(term_coefficients))))
    max_shots_per_term = min(n_shots, max_shots_per_term)  # Don't let max_shots_per_term > n_shots if manually defined

    n_terms = len(grouped_terms)
    shot_allocation = np.zeros(n_terms, dtype=int)

    while True:
        remaining_shots = n_shots - sum(shot_allocation)
        if not remaining_shots:
            break

        # In case n_shots is too high s.t. max_shots_per_term is too low
        # Increase max_shots_per_term, and redo the shot allocation
        if np.min(shot_allocation) == max_shots_per_term:
            max_shots_per_term = min(2 * max_shots_per_term, n_shots)
            shot_allocation = np.zeros(n_terms, dtype=int)
            continue

        if method in ("c", "coefficients"):
            # Split shots based on the relative magnitudes of the coefficients of the (group of) Pauli term(s)
            # and only for terms that haven't reached the upper limit yet
            term_coefficients = np.array(
                [
                    (
                        sum(abs(coeff) for _t, coeff in expression.as_coefficients_dict().items())
                        if shots < max_shots_per_term
                        else 0.0
                    )
                    for shots, (expression, _) in zip(shot_allocation, grouped_terms)
                ],
                dtype=float,
            )
            # Normalise term_coefficients, then get an initial distribution of remaining_shots
            term_coefficients /= sum(term_coefficients)
            _shot_allocation = (remaining_shots * term_coefficients).astype(int)
            # Only keep the terms with >0 shots allocated, renormalise term_coefficients, and distribute again
            term_coefficients *= _shot_allocation > 0
            if _shot_allocation.any():
                term_coefficients /= sum(term_coefficients)
                _shot_allocation = (remaining_shots * term_coefficients).astype(int)
            else:
                # For distributing the remaining few shots, i.e. remaining_shots << n_terms
                _shot_allocation = np.array(
                    allocate_shots(grouped_terms, remaining_shots, max_shots_per_term=remaining_shots, method="u")
                )

        elif method in ("u", "uniform"):
            # Uniform distribution of shots for every term. Extra shots are randomly distributed
            _shot_allocation = np.array([remaining_shots // n_terms for _ in range(n_terms)])
            if not _shot_allocation.any():
                _shot_allocation = np.zeros(n_terms)
                _shot_allocation[:remaining_shots] = 1
                np.random.shuffle(_shot_allocation)

        else:
            raise NameError("Unknown method!")

        # Add on to the current allocation, and set upper limit to the number of shots for a given term
        shot_allocation += _shot_allocation.astype(int)
        shot_allocation = np.clip(shot_allocation, 0, max_shots_per_term)

    return shot_allocation.tolist()
