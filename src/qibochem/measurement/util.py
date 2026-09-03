"""
Utility functions for optimising measurements and calculation of expectation value
"""

import networkx as nx
import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z
from sympy.core.expr import Expr
from sympy.core.numbers import One

# Mapping of Pauli operators to a symplectic (binary) representation, folowing the convention of (X|Z)
PAULI_BINARY = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
BINARY_PAULI = {symplectic: pauli for pauli, symplectic in PAULI_BINARY.items()}

SYMPLECTIC_PHASE_TABLE = [1.0, 1.0j, -1.0j]
SYMPLECTIC_INDEX = {symplectic: index for index, symplectic in enumerate(BINARY_PAULI.keys())}


def _pauli_to_symplectic(pauli_term: Expr, nqubits: int) -> np.ndarray:
    """
    Map a single Pauli term to the corresponding symplectic vector ((1D np.ndarray)).
    `nqubits` is the number of qubits used for the molecular Hamiltonian; needed to define dimensions of the vector
    """
    # Pauli operator for each qubit
    pauli_ops = (
        {pauli_op.target_qubit: str(pauli_op)[0] for pauli_op in pauli_term.args if isinstance(pauli_op, (X, Y, Z))}
        if pauli_term.args
        else {pauli_term.target_qubit: str(pauli_term)[0]}
    )
    # Convert to the symplectic vector
    sym_vector = np.reshape(
        np.array([PAULI_BINARY[pauli_ops.get(i, "I")] for i in range(nqubits)], dtype=np.uint8),
        shape=2 * nqubits,
        order="F",
    )
    return sym_vector


def _check_terms_commutativity(term1: np.ndarray, term2: np.ndarray, qubitwise: bool) -> bool:
    """
    Check if terms 1 and 2 (in symplectic form) are mutually commuting. The 'qubitwise' argument determines if the
    check is for general commutativity (False), or the stricter qubitwise commutativity.
    """
    term1_nqubits = term1.shape[0] // 2
    term2_nqubits = term2.shape[0] // 2
    nqubits = min(term1_nqubits, term2_nqubits)  # Only compare common qubits
    if qubitwise:
        # Qubitwise condition: x1z2 + x2z1 == 0
        return all(
            ((term1[i] & term2[i + term2_nqubits]) ^ (term1[i + term1_nqubits] & term2[i])) == 0 for i in range(nqubits)
        )
    # General commutativity: Even number of anti-commuting operators
    n_noncommuting_ops = sum(
        (term1[i] & term2[i + term2_nqubits]) ^ (term1[i + term1_nqubits] & term2[i]) for i in range(nqubits)
    )
    return n_noncommuting_ops % 2 == 0


def _graph_colouring(terms_dict: dict[Expr, tuple[float, np.ndarray]], qubitwise: bool) -> list[list[Expr]]:
    """
    Groups Pauli terms by solving the minimum clique cover (i.e. as few cliques as possible) problem for the graph
    whereby each node is a Pauli string, and an edge exists between two nodes iff they commute. This is equivalent to
    the graph colouring problem of the complement graph (i.e. edge between nodes if they DO NOT commute), which this
    function follows.

    Args:
        terms_dict (dict[Expr, tuple[float, np.ndarray]]): Pauli terms to be grouped; given as a dict whereby the keys
            are the Pauli terms (Expr), and their corresponding values are two-tuples: term coefficient and the
            symplectic form of the Pauli term respectively.
        qubitwise (bool): Determines if the check is for general commutativity or the stricter qubitwise commutativity

    Returns:
        list[list[Expr]]: Groups (lists) of Pauli strings that mutually commute within each group
    """
    G = nx.Graph()
    # Complement graph: Add all the terms as nodes first, then add edges between nodes if they DO NOT commute
    G.add_nodes_from(terms_dict)
    G.add_edges_from(
        (term1, term2)
        for i1, term1 in enumerate(terms_dict)
        for i2, term2 in enumerate(terms_dict)
        if i2 > i1 and not _check_terms_commutativity(terms_dict[term1][1], terms_dict[term2][1], qubitwise)
    )
    # Solve using Greedy Colouring on NetworkX
    sorted_groups = nx.coloring.greedy_color(G)
    group_ids = set(sorted_groups.values())
    # Sort results so that test results will be replicable
    term_groups = [[group for group, group_id in sorted_groups.items() if group_id == _id] for _id in group_ids]
    return term_groups


def _sorted_insertion(terms_dict: dict[Expr, tuple[float, np.ndarray]], qubitwise: bool) -> list[list[Expr]]:
    """
    Groups Pauli terms by sorting the terms w.r.t. their coefficients (largest first). For each of the sorted terms, if
    it is compatible with an existing group, allocate it there; otherwise, assign it to a new group.

    Args:
        terms_dict (dict[Expr, tuple[float, np.ndarray]]): Pauli terms to be grouped; given as a dict whereby the keys
            are the Pauli terms (Expr), and their corresponding values are two-tuples: term coefficient and the
            symplectic form of the Pauli term respectively.
        qubitwise (bool): Determines if the check is for general commutativity or the stricter qubitwise commutativity

    Returns:
        list[list[Expr]]: Groups (lists) of Pauli strings that mutually commute within each group
    """
    sorted_terms = sorted(terms_dict, key=lambda x: abs(terms_dict[x][0]), reverse=True)
    term_groups = []
    for term in sorted_terms:
        added = False
        for group in term_groups:
            # Check if current term commutes with all terms in current group
            if all(_check_terms_commutativity(terms_dict[term][1], terms_dict[_term][1], qubitwise) for _term in group):
                group.append(term)
                added = True
                break
        if not added:
            term_groups.append([term])
    return term_groups


def _group_commuting_terms(
    hamiltonian: SymbolicHamiltonian, qubitwise: bool, method: str = "sorted"
) -> list[list[Expr]]:
    """
    Groups Pauli terms in hamiltonian into groups of (possibly qubitwise) commuting terms

    Args:
        hamiltonian (SymbolicHamiltonian): Hamiltonian to be sorted into groups of commuting terms
        qubitwise (bool): Determines if the check is for general commutativity, or the stricter qubitwise commutativity
        method (str): Method used to group the Pauli terms. Must be either "sorted" (default) or "graph". More details
            on both methods are given in their respective functions

    Returns:
        list[list[str]]: Containing groups (lists) of Pauli strings that all commute mutually
    """
    terms_dict = {
        term: (coeff, _pauli_to_symplectic(term, hamiltonian.nqubits))
        for term, coeff in hamiltonian.form.as_coefficients_dict().items()
        if not isinstance(term, One)
    }

    term_groups = []
    if method == "sorted":
        term_groups = _sorted_insertion(terms_dict, qubitwise)
    elif method == "graph":
        term_groups = _graph_colouring(terms_dict, qubitwise)
    else:
        raise_error(ValueError, "Invalid method argument for grouping commuting terms!")
    return term_groups


def _symplectic_to_pauli(symplectic_vector: np.ndarray) -> list[str]:
    """Map a single symplectic vector to its corresponding Pauli term (E.g. ['Y0', 'X2'])"""
    dim = symplectic_vector.shape[0] // 2
    pauli_op_vectors = [tuple(symplectic_vector[[_i, _i + dim]]) for _i in range(dim)]
    pauli_op_terms = [
        f"{BINARY_PAULI[vector]}{_q}"
        for _q, vector in zip(range(dim), pauli_op_vectors)
        if vector != (0, 0)  # Not retaining I terms
    ]
    return pauli_op_terms


def _binary_gaussian_elimination(vector_space: np.ndarray) -> np.ndarray:
    """
    Performs Gaussian elimination on a binary vector_space. Returns the (unique) reduced row echelon form, and removes
    any zero rows as well
    """
    vector_space = np.array(vector_space, dtype=np.uint8)  # Create a copy for returning
    rows, cols = vector_space.shape

    pivot_row = 0
    for col in range(cols):
        # Find a pivot row with a 1 in current column.
        pivot_candidates = np.where(vector_space[pivot_row:, col] == 1)[0]
        if pivot_candidates.size == 0:
            continue

        row = pivot_row + pivot_candidates[0]

        # Swap current row with pivot row if needed.
        if pivot_row != row:
            vector_space[[row, pivot_row]] = vector_space[[pivot_row, row]]

        # Eliminate all other rows
        rows_to_reduce = np.where(vector_space[:, col] == 1)[0]
        rows_to_reduce = rows_to_reduce[rows_to_reduce != pivot_row]

        # In GF(2), elimination is XOR with the pivot row.
        vector_space[rows_to_reduce] ^= vector_space[pivot_row]

        pivot_row += 1
        if pivot_row == rows:
            break

    # Remove all zero rows from the obtained basis
    zero_vector_indices = np.all(vector_space == 0, axis=1)
    vector_space = vector_space[~zero_vector_indices]
    return vector_space


def _binary_nullspace(binary_matrix: np.ndarray) -> np.ndarray:
    """Finds the nullspace of a binary_matrix, i.e. x s.t. Ax = 0"""
    dim = binary_matrix.shape[0]
    # Form the augmented matrix
    aug_matrix = np.concatenate((binary_matrix.T, np.identity(binary_matrix.shape[1], dtype=np.uint8)), axis=1)
    rref_aug_matrix = _binary_gaussian_elimination(aug_matrix)
    nullspace = rref_aug_matrix[dim:, dim:]
    return nullspace


def _symplectic_inner_product(u: np.ndarray, v: np.ndarray) -> int:
    """
    Inner product of the symplectic vector space := (u, Jv), where J = [[0_{NxN}, I_{NxN}], [I_{NxN}, 0_{NxN}]].
    Returns 0 or 1, where 0 means that u commutes with v, and 1 implies that they do not commute
    """
    dim = u.shape[0] // 2
    return (np.dot(u[:dim], v[dim:]) + np.dot(u[dim:], v[:dim])) % 2


def _lagrangian_subspace(vector_space: np.ndarray) -> np.ndarray:
    """Find Lagrangian subspace of the given vector space; the symplectic nullspace in this context"""
    # Remove rows from cp_vector_space until cp_vector_space.shape matches (N, 2N)
    while vector_space.shape[0] > (vector_space.shape[1] // 2):
        anticommuting_vector_indices, anticommuting_vectors = None, None
        # Find a pair of anti-commuting vectors in vector_space
        for i1, v1 in enumerate(vector_space):
            for i2, v2 in enumerate(vector_space):
                if i2 > i1 and _symplectic_inner_product(v1, v2) == 1:
                    anticommuting_vector_indices = [i1, i2]
                    anticommuting_vectors = vector_space[anticommuting_vector_indices]
                    break
            if anticommuting_vector_indices is not None:
                break

        # Remove the two anti-commuting vectors from the basis
        space_to_orthogonalize = np.delete(vector_space, anticommuting_vector_indices, axis=0)
        for i1, vector in enumerate(space_to_orthogonalize):
            for i2, anticommuting_vector in enumerate(anticommuting_vectors):
                space_to_orthogonalize[i1] ^= (
                    _symplectic_inner_product(vector, anticommuting_vectors[1 - i2]) * anticommuting_vector
                )

        # Preferentially select Z over X
        first_nonzero_col = np.argmax(anticommuting_vectors, axis=1)
        selected_vector = anticommuting_vectors[np.argmax(first_nonzero_col)]

        vector_space = np.append([selected_vector], space_to_orthogonalize, axis=0)
        vector_space = _binary_gaussian_elimination(vector_space)

    return vector_space


def _sort_tau_terms(v_basis: np.ndarray) -> np.ndarray:
    """Sorts the rows of v_basis s.t. the (i, i) and (i, i+dim) entries are not 0, i.e. i'th basis vector i is NOT I"""
    dim = v_basis.shape[0]
    while not all(v_basis[i, i] or v_basis[i, i + dim] for i in range(dim)):
        # Sort unmatched qubits
        unmatched_qubits = [i for i in range(dim) if not (v_basis[i, i] or v_basis[i, i + dim])]
        matches_for_unmatched_qubits = {
            i: [qubit for qubit in range(dim) if v_basis[i, qubit] or v_basis[i, qubit + dim]] for i in unmatched_qubits
        }
        # Preference: Qubits with fewest candidates (tie-break: min(qubit index))
        row_to_swap = min(matches_for_unmatched_qubits, key=lambda x: (len(matches_for_unmatched_qubits[x]), x))
        target = min(matches_for_unmatched_qubits[row_to_swap])
        v_basis[[row_to_swap, target]] = v_basis[[target, row_to_swap]]
    return v_basis


def _get_sigma_terms(tau_terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the set of sigma terms for a given array of tau terms's, with (sigma_i|tau_j) = 1 if i == j else 0, and
    (sigma_i|sigma_j) == 0 if i != j, i.e. all sigma_i's must correspond to different qubits. Note that tau_terms is
    also re-orthogonalised to follow the first relation given above in the process.
    """
    sigma_terms = []
    dim = tau_terms[0].shape[0] // 2
    # Make a copy of the original basis set for orthogonalization
    new_tau_terms = np.array(tau_terms, dtype=np.uint8)
    # Iterate over the original tau_i to make changes to new_tau_i
    for _i in range(dim):
        tau_i = new_tau_terms[_i]
        # Let sigma_i be x_i if z_i is in tau_i, otherwise let sigma_i be z_i
        _sigma_i = (0, 1) if tuple(tau_i[[_i, _i + dim]].tolist()) != (0, 1) else (1, 0)
        # Convert and broadcast _sigma_i back to the correct size using I's
        sigma_i = np.ravel(np.array([(0, 0) if _j != _i else _sigma_i for _j in range(dim)]).T)
        sigma_terms.append(sigma_i)
        # Orthogonalise the non-i^th terms:
        new_tau_terms ^= np.array(
            [
                _symplectic_inner_product(new_tau_terms[_j], sigma_i) * tau_i if _j != _i else np.zeros(2 * dim)
                for _j in range(dim)
            ],
            dtype=np.uint8,
        )
    return new_tau_terms, np.array(sigma_terms, dtype=np.uint8)


def _solve_linear_system(binary_matrix: np.ndarray, vector: np.ndarray) -> list[np.ndarray]:
    """Solve (binary) linear system Ax = b. Each item in the result corresponds to the respective vectors in b"""
    # Form the augmented matrix and row-reduce it using Gaussian elimination
    aug_matrix = np.concatenate((binary_matrix, vector), axis=0).T
    rref_aug_matrix = _binary_gaussian_elimination(aug_matrix)
    # Get non-zero entries in each column on RHS of rref_aug_matrix => Solution for respective vector in b
    return [np.nonzero(rref_aug_matrix[:, binary_matrix.shape[0] + i])[0].tolist() for i in range(vector.shape[0])]


def _single_qubit_phase_factor(pauli_ops: list[np.ndarray]) -> complex:
    """Compute the phase factor w.r.t. the product of multiple Pauli operators for a single qubit"""
    # Initialise as 1.0*I, then multiply with each Pauli operator acting on that qubit
    coeff, current_pauli_op = 1.0, np.zeros(2)
    for pauli_op in pauli_ops:
        # If I, just skip
        if SYMPLECTIC_INDEX[tuple(current_pauli_op)] == 0:
            current_pauli_op = pauli_op
            continue
        if SYMPLECTIC_INDEX[tuple(pauli_op)] == 0:
            continue
        # Multiply by some phase factor depending on what Pauli operators are involved
        coeff *= SYMPLECTIC_PHASE_TABLE[SYMPLECTIC_INDEX[tuple(pauli_op)] - SYMPLECTIC_INDEX[tuple(current_pauli_op)]]
        current_pauli_op = (current_pauli_op + pauli_op) % 2
    return coeff


def _phase_factor(pauli_terms: list[np.ndarray]) -> int:
    """Compute phase factor of a product of mutually commuting Pauli terms (in symplectic form). Returns: 1 or -1"""
    # Singleton case is trivial: 1
    if len(pauli_terms) == 1:
        return 1
    # >1 term:
    dim = pauli_terms[0].shape[0] // 2
    coefficient = 1.0
    for qubit in range(dim):
        # Get all Pauli operators for a particular qubit
        pauli_ops = [pauli_term[[qubit, qubit + dim]] for pauli_term in pauli_terms]
        coefficient *= _single_qubit_phase_factor(pauli_ops)
    return int(np.real_if_close(coefficient))


def _make_x_matrix_full_rank(stabiliser_matrix: np.ndarray, phases: np.ndarray) -> list[gates.Gate]:
    """
    Modifies stabiliser_matrix and phases in-place to transform 'X matrix' to full rank, with H gates representing each 'swap'
    of columns between the 'Z' and 'X' matrices. Note: stabiliser_matrix should already be in reduced row echelon form

    Returns:
        list[gates.Gate]: List of H gates to be added to the circuit
    """
    gates_list = []

    dim_space = stabiliser_matrix.shape[1] // 2
    x_matrix = stabiliser_matrix[:, :dim_space]
    z_matrix = stabiliser_matrix[:, dim_space:]

    # Need to find full rank submatrix in Z matrix for each of the zero rows in the X matrix
    qubits = []
    zero_row_indices = np.where(np.all(x_matrix == 0, axis=1))[0]
    while zero_row_indices.size > 0:
        # Select the first possible column for the first zero row
        for qubit in np.nonzero(z_matrix[zero_row_indices[0], :])[0]:
            if qubit not in qubits:
                # For S(a)/H(a): r_i := r_i + x_{i,a} z_{i,a} for all i
                phases ^= stabiliser_matrix[:, qubit] * stabiliser_matrix[:, qubit + dim_space]
                stabiliser_matrix[:, [qubit, qubit + dim_space]] = stabiliser_matrix[:, [qubit + dim_space, qubit]]
                gates_list.append(gates.H(qubit))
                qubits.append(qubit)
                break
        zero_row_indices = np.where(np.all(x_matrix == 0, axis=1))[0]
    return gates_list


def _col_reduce_x_matrix(stabiliser_matrix: np.ndarray, phases: np.ndarray) -> list[gates.Gate]:
    """
    Modifies stabiliser_matrix and phases in-place to transform the X matrix to I, using CNOT/SWAP gates

    Returns:
        list[gates.Gate]: List of CNOT/SWAP gates to be added to the circuit
    """
    gates_list = []
    dim, dim_space = stabiliser_matrix.shape
    dim_space = dim_space // 2

    pivot_col = 0
    # Paper used row reduction, but should be column reduction in our context
    for row in range(dim):
        if pivot_col >= dim_space:
            break
        # Get columns at row i with 1
        nonzero_cols = np.where(stabiliser_matrix[row, pivot_col:dim_space] == 1)[0]

        col = pivot_col + nonzero_cols[0]

        # Move pivot column of X matrix into position
        if col != pivot_col:
            stabiliser_matrix[:, [pivot_col, col, pivot_col + dim_space, col + dim_space]] = stabiliser_matrix[
                :, [col, pivot_col, col + dim_space, pivot_col + dim_space]
            ]
            gates_list.append(gates.SWAP(col, pivot_col))

        # Eliminate other 1's in the present row
        nonzero_cols = np.where(stabiliser_matrix[row, :dim_space] == 1)[0]
        nonzero_cols = nonzero_cols[nonzero_cols != pivot_col]

        # Remove all nonzero entries on row _i using CNOT gates
        for col in nonzero_cols:
            # For CNOT(a, b): r_i := r_i + x_{i,a} z_{i,b} (x_{i,b} + z_{i,a} + 1), for all i
            phase_changes = (
                stabiliser_matrix[:, pivot_col]
                & stabiliser_matrix[:, col + dim_space]
                & (stabiliser_matrix[:, col] ^ stabiliser_matrix[:, pivot_col + dim_space] ^ 1)
            )
            phases ^= phase_changes
            # X matrix: Add pivot column to column with 1
            stabiliser_matrix[:, col] ^= stabiliser_matrix[:, pivot_col]
            # Z matrix: Add (column with 1)^th column to pivot column
            stabiliser_matrix[:, pivot_col + dim_space] ^= stabiliser_matrix[:, col + dim_space]
            gates_list.append(gates.CNOT(pivot_col, col))
        pivot_col += 1

    return gates_list


def _zero_z_matrix(stabiliser_matrix: np.ndarray, phases: np.ndarray) -> list[gates.Gate]:
    """
    Modifies stabiliser_matrix and phases in-place to transform the Z matrix to a zero matrix.
    1. S gates used to set diagonal entries on Z matrix
    2. CZ gates used to remove off-diagonal entries on Z matrix (Phases not updated)

    Returns:
        list[gates.Gate]: List of S and CZ gates to be added to the circuit
    """
    s_gates = []
    cz_gates = []
    dim, dim_space = stabiliser_matrix.shape
    dim_space = dim_space // 2
    # Following the algorithm in the paper, zero out the diagonal entries first
    for i in range(dim):
        if stabiliser_matrix[i, dim_space + i] == 1:
            # For S(a)/H(a): r_i := r_i + x_{i,a} z_{i,a} for all i
            phases ^= stabiliser_matrix[:, i] * stabiliser_matrix[:, i + dim_space]
            stabiliser_matrix[i, dim_space + i] = 0
            s_gates.append(gates.S(i))
        # Then remove the off-diagonal terms in each row
        for j in range(dim_space):
            if j > i and stabiliser_matrix[i, dim_space + j] == 1:
                # Note: Not updating phases here
                stabiliser_matrix[i, dim_space + j] = 0
                stabiliser_matrix[j, dim_space + i] = 0
                cz_gates.append(gates.CZ(i, j))
    return s_gates + cz_gates


def _synthesise_circuit(v_basis: np.ndarray) -> tuple[list[gates.Gate], list[int]]:
    """
    Gets the basis rotation gates for rotating the initial measurement basis into the computational basis.
    The stabiliser matrix (v_basis) follows the format of (X|Z) matrices.

    Returns:
        list[gates.Gate]: Gates to be added after the circuit ansatz
        list[int]: Phases of the measured basis terms
    """
    stabiliser_matrix = np.array(v_basis, dtype=np.uint8)
    nqubits = stabiliser_matrix.shape[0]
    phases = np.array([[0 for _ in range(nqubits)]], dtype=np.uint8)  # To keep track of phases
    rotation_gates = []
    # 1. Apply H gates to transform 'X matrix' to full rank
    rotation_gates += _make_x_matrix_full_rank(stabiliser_matrix, phases)
    # 2. Row-reduce 'X matrix' to I using CNOT/SWAP gates
    rotation_gates += _col_reduce_x_matrix(stabiliser_matrix, phases)
    # 3. Remove all non-zero entries on 'Z matrix' using S and CZ gates
    rotation_gates += _zero_z_matrix(stabiliser_matrix, phases)
    # 4. Apply H to each qubit to swap the 'X' and 'Z' matrices. Note: Not gonna update phases here
    rotation_gates += [gates.H(i) for i in range(nqubits)]
    # Update circuit phase factors to be 1 or -1
    phases = [-1 if x else 1 for x in phases[0]]
    return rotation_gates, phases
