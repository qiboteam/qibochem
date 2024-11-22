"""
Utility functions for optimising measurements and calculation of expectation value
"""

import networkx as nx
import numpy as np

# Mapping of Pauli operators to a symplectic (binary) representation, folowing the convention of (X|Z)
PAULI_BINARY = {"I": (0, 0), "Z": (0, 1), "X": (1, 0), "Y": (1, 1)}
BINARY_PAULI = {(0, 0): "I", (0, 1): "Z", (1, 0): "X", (1, 1): "Y"}  # Vice versa


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


def pauli_to_symplectic(pauli_string, n_qubits):
    """
    Map a single Pauli term to the corresponding symplectic vector

    Args:
        pauli_string: Iterable of strings representing a single Pauli term, e.g ["X0", "Y26", "Z200"]
            Can get with: [factor.name for factor in term.factors], where term is a Qibo SymbolicTerm
        n_qubits: Number of qubits used for the molecular Hamiltonian; needed to define the dimensions of the vector

    Returns:
        np.array: Symplectic vector for the given Pauli string (1D np.array)
    """
    # Parse the Pauli string to return the single qubit Pauli operator for each qubit
    pauli_ops = {int(pauli_op[1:]): pauli_op[0] for pauli_op in pauli_string}
    # Convert to the symplectic vector
    sym_vector = np.reshape(
        np.array([PAULI_BINARY[pauli_ops.get(_i, "I")] for _i in range(n_qubits)]), newshape=2 * n_qubits, order="F"
    )
    return sym_vector


def symplectic_to_pauli(symplectic_vector):
    """
    Map a symplectic vector back to a single Pauli term

    Args:
        symplectic_vector (np.array): Symplectic vector to be converted

    Returns:
        list: A list of Pauli operators for a single Pauli term, e.g. ['Y0', 'X2']
    """
    dim = symplectic_vector.shape[0] // 2

    pauli_op_vectors = [tuple(symplectic_vector[[_i, _i + dim]]) for _i in range(dim)]
    pauli_op_terms = [
        f"{BINARY_PAULI[vector]}{_q}"
        for _q, vector in zip(range(dim), pauli_op_vectors)
        if vector != (0, 0)  # Not retaining I terms
    ]
    return pauli_op_terms


def symplectic_inner_product(u, v):
    """
    Inner product of the symplectic vector space := (u, Jv), where
    J = [[0_{NxN}, I_{NxN}], [I_{NxN}, 0_{NxN}]]

    Returns:
        0 or 1, where 0 means that u commutes with v, and 1 implies that they do not commute"""
    dim = u.shape[0] // 2
    return (np.dot(u[:dim], v[dim:]) + np.dot(u[dim:], v[:dim])) % 2
