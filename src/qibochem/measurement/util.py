"""
Utility functions for optimising measurements and calculation of expectation value
"""

import networkx as nx
import numpy as np
from qibo import gates

# Mapping of Pauli operators to a symplectic (binary) representation, folowing the convention of (X|Z)
PAULI_BINARY = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
BINARY_PAULI = {symplectic: pauli for pauli, symplectic in PAULI_BINARY.items()}

SYMPLECTIC_PHASE_TABLE = [1.0, 1.0j, -1.0j]
SYMPLECTIC_VECTORS = [[0, 0], [1, 0], [1, 1], [0, 1]]


def _get_qubit(pauli_op: str) -> int:
    """Extract the qubit index from a Pauli operator, e.g. "X12" -> 12"""
    return int(pauli_op[1:])


def check_terms_commutativity(term1: str, term2: str, qubitwise: bool) -> bool:
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
    common_qubits = {_get_qubit(_op) for _op in term1.split() if _op[0] != "I"} & {
        _get_qubit(_op) for _op in term2.split() if _op[0] != "I"
    }
    if not common_qubits:
        return True
    # Get the single Pauli operators for the common qubits for both Pauli terms
    term1_ops = [_op for _op in term1.split() if _get_qubit(_op) in common_qubits]
    term2_ops = [_op for _op in term2.split() if _get_qubit(_op) in common_qubits]
    if qubitwise:
        # Qubitwise: Compare the Pauli terms at the common qubits. Any difference => False
        return all(_op1 == _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # General commutativity:
    # Get the number of single Pauli operators that do NOT commute
    n_noncommuting_ops = sum(_op1 != _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # term1 and term2 have general commutativity iff n_noncommuting_ops is even
    return n_noncommuting_ops % 2 == 0


def group_commuting_terms(terms_list: list[str], qubitwise: bool) -> list[str]:
    """
    Groups the terms in terms_list into as few groups as possible, where all the terms in each group commute
    mutually == Finding the minimum clique cover (i.e. as few cliques as possible) for the graph whereby each node
    is a Pauli string, and an edge exists between two nodes iff they commute.

    This is equivalent to the graph colouring problem of the complement graph (i.e. edge between nodes if they DO NOT
    commute), which this function follows.

    Args:
        terms_list (List(str)): List of strings. The strings should follow the output from
            ``" ".join(factor.name for factor in term.factors)``, where term is a Qibo SymbolicTerm. E.g. "X0 Z1".
        qubitwise (bool): Determines if the check is for general commutativity, or the stricter qubitwise commutativity

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


def pauli_to_symplectic(pauli_string: list[str], n_qubits: int) -> np.ndarray:
    """
    Map a single Pauli term to the corresponding symplectic vector

    Args:
        pauli_string: Iterable of strings representing a single Pauli term, e.g ["X0", "Y26", "Z200"]
            Can get with: [factor.name for factor in term.factors], where term is a Qibo SymbolicTerm
        n_qubits: Number of qubits used for the molecular Hamiltonian; needed to define the dimensions of the vector

    Returns:
        np.array: Symplectic vector for the given Pauli string (1D np.array)
    """
    pauli_ops = {_get_qubit(pauli_op): pauli_op[0] for pauli_op in pauli_string}  # Pauli operator for each qubit
    # Convert to the symplectic vector
    sym_vector = np.reshape(
        np.array([PAULI_BINARY[pauli_ops.get(_i, "I")] for _i in range(n_qubits)]), shape=2 * n_qubits, order="F"
    )
    return sym_vector


def symplectic_to_pauli(symplectic_vector: np.ndarray) -> list[str]:
    """
    Map a single symplectic vector back to a single Pauli term

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


def symplectic_inner_product(u: np.ndarray, v: np.ndarray) -> int:
    """
    Inner product of the symplectic vector space := (u, Jv), where
    J = [[0_{NxN}, I_{NxN}], [I_{NxN}, 0_{NxN}]]

    Returns:
        int: 0 or 1, where 0 means that u commutes with v, and 1 implies that they do not commute
    """
    dim = u.shape[0] // 2
    return (np.dot(u[:dim], v[dim:]) + np.dot(u[dim:], v[:dim])) % 2


def binary_gaussian_elimination(vector_space: np.ndarray) -> np.ndarray:
    """
    Carries out Gaussian elimination on a binary vector_space to obtain a basis for vector_space. Reduces vector_space
    to its (unique) reduced row echelon form, and removes any zero rows as well

    Args:
        vector_space (np.ndarray): Binary vector space

    Returns:
        np.ndarray: Basis set for vector_space
    """
    cp_vector_space = np.array(vector_space)

    dim = vector_space.shape[0]
    # Swap the rows in the vector space to get its row echelon form
    for _i in range(dim):
        subspace_to_sort = cp_vector_space[_i:, :]
        if not np.any(subspace_to_sort):
            break

        # Always take the first nonzero column to sort
        nonzero_cols = np.nonzero(np.any(subspace_to_sort, axis=0))[0]
        _col = nonzero_cols[0]

        col_indices = subspace_to_sort[:, _col].argsort()[::-1]
        subspace_to_sort[:, :] = subspace_to_sort[col_indices]

        # Other than row _i, find which rows have 1 in column _i
        rows_to_reduce = [_j for _j, _row in enumerate(cp_vector_space) if _j != _i and cp_vector_space[_j, _col] == 1]

        # Add row _i to each of the rows with 1 in the same column
        cp_vector_space[[rows_to_reduce]] += cp_vector_space[_i]
        cp_vector_space %= 2

    # Remove all zero rows from the obtained basis
    zero_vector_indices = np.all(cp_vector_space == 0, axis=1)
    cp_vector_space = cp_vector_space[~zero_vector_indices]
    return cp_vector_space


def binary_nullspace(binary_matrix: np.ndarray) -> np.ndarray:
    """Finds the nullspace of a binary_matrix, i.e. x s.t. Ax = 0"""
    dim = binary_matrix.shape[0]
    # Form the augmented matrix
    aug_matrix = np.concatenate((binary_matrix.T, np.identity(binary_matrix.shape[1])), axis=1)
    rref_aug_matrix = binary_gaussian_elimination(aug_matrix)
    nullspace = rref_aug_matrix[dim:, dim:]
    return nullspace.astype(int)


def lagrangian_subspace(vector_space: np.ndarray) -> np.ndarray:
    """
    Find the Lagrangian subspace of some vector space; the symplectic nullspace in this context

    Returns:
        np.ndarray: Basis vectors of vector_space. Shape of the array should be (N, 2N)
    """
    cp_vector_space = np.array(vector_space)
    # While loop to remove rows from cp_vector_space until cp_vector_space.shape matches (N, 2N)
    while True:
        anticommuting_vector_indices = None
        # Find a pair of anti-commuting vectors in vector_space
        for _i1, _v1 in enumerate(cp_vector_space):
            for _i2, _v2 in enumerate(cp_vector_space):
                if _i2 > _i1 and symplectic_inner_product(_v1, _v2) == 1:
                    anticommuting_vector_indices = [_i1, _i2]
                    anticommuting_vectors = cp_vector_space[anticommuting_vector_indices]
                    break

        if cp_vector_space.shape[0] == (cp_vector_space.shape[1] // 2):
            break

        # Remove the two anti-commuting vectors from the basis
        space_to_orthogonalize = np.delete(cp_vector_space, anticommuting_vector_indices, axis=0)
        for _i1, vector in enumerate(space_to_orthogonalize):
            for _i2, anticommuting_vector in enumerate(anticommuting_vectors):
                space_to_orthogonalize[_i1] += (
                    symplectic_inner_product(vector, anticommuting_vectors[1 - _i2]) * anticommuting_vector
                )
                space_to_orthogonalize = space_to_orthogonalize % 2

        cp_vector_space = np.append([anticommuting_vectors[0]], space_to_orthogonalize, axis=0)
        cp_vector_space = binary_gaussian_elimination(cp_vector_space)

    return cp_vector_space


def sort_tau_terms(v_basis: np.ndarray) -> np.ndarray:
    """
    Sorts v_basis s.t. the i'th term of basis vector i is NOT I, e.g.
    [['X0', 'X2'], ['Z1', 'X3', 'Z4', 'X5'], ['Z0', 'Z2'], ['Z1'], ['Z3', 'Z5'], ['Z4']]
    will return
    [['X0', 'X2'], ['Z1'], ['Z0', 'Z2'], ['Z3', 'Z5'], ['Z4'], ['Z1', 'X3', 'Z4', 'X5']]

    Returns:
        np.ndarray: Sorted array of basis vectors that can be used for finding sigma_i directly
    """
    # Convert the basis set to strings for easier sorting
    pauli_terms = [symplectic_to_pauli(vector) for vector in v_basis]
    dim = len(pauli_terms)
    sorted_terms = {}
    remaining = list(pauli_terms)

    while remaining:
        qubit_terms = {
            qubit: [term for term in remaining if any(_get_qubit(_op) == qubit for _op in term)]
            for qubit in range(dim)
            if qubit not in sorted_terms
        }
        # Preference: Qubits with fewest candidates (tie-break: min(qubit index))
        qubit = min(qubit_terms, key=lambda x: (len(qubit_terms[x]), x))
        selected_term = min(qubit_terms[qubit], key=len)
        sorted_terms[qubit] = selected_term
        remaining.remove(selected_term)
    # Convert the strings back to symplectic vectors and return the whole array
    return np.array([pauli_to_symplectic(sorted_terms[_i], dim) for _i in range(dim)])


def get_sigma_terms(tau_terms: np.ndarray) -> tuple:
    """
    Find the set of sigma terms for a given array of tau terms's, with (sigma_i|tau_j) = 1 if i == j else 0, and
    (sigma_i|sigma_j) == 0 if i != j, i.e. all sigma_i's must correspond to different qubits. Note that tau_terms must
    also be re-orthogonalised to follow the first relation given above in the process.

    Args:
        tau_terms: Basis set of the V subspace

    Returns:
        np.array, np.array: The new (re-orthogonalized) tau_i terms, and the sigma_i terms
    """
    sigma_terms = []
    dim = tau_terms[0].shape[0] // 2
    # Make a copy of the original basis set for orthogonalization
    new_tau_terms = np.array(tau_terms)
    # Iterate over the original tau_i to make changes to new_tau_i
    for _i in range(dim):
        tau_i = new_tau_terms[_i]
        # Let sigma_i be x_i if z_i is in tau_i, otherwise let sigma_i be z_i
        _sigma_i = (0, 1) if tuple(tau_i[[_i, _i + dim]].tolist()) != (0, 1) else (1, 0)
        # Convert and broadcast _sigma_i back to the correct size using I's
        sigma_i = np.ravel(np.array([(0, 0) if _j != _i else _sigma_i for _j in range(dim)]).T)
        sigma_terms.append(sigma_i)
        # Orthogonalise the non-i^th terms:
        new_tau_terms += np.array(
            [
                # Not sure if need _j != _i or if _j > _i is good enough?
                # Paper says do _j > _i, but then will have some non-commuting tau/sigma's...?
                symplectic_inner_product(new_tau_terms[_j], sigma_i) * tau_i if _j != _i else np.zeros(2 * dim)
                # symplectic_inner_product(new_tau_terms[_j], sigma_i) * tau_i if _j > _i else np.zeros(2 * dim)
                for _j in range(dim)
            ]
        ).astype(int)
        new_tau_terms = new_tau_terms % 2

    return new_tau_terms, np.array(sigma_terms)


def solve_linear_system(binary_matrix: np.ndarray, vector: np.ndarray) -> list[np.ndarray]:
    """
    Solve the (binary) linear system Ax = b

    Returns:
        list: Each item in the list corresponds to the respective vectors in b.
    """
    # Form the augmented matrix and row-reduce it using Gaussian elimination
    aug_matrix = np.concatenate((binary_matrix, vector), axis=0).T
    rref_aug_matrix = binary_gaussian_elimination(aug_matrix)
    # Get non-zero entries in each column on RHS of rref_aug_matrix => Solution for respective vector in b
    return [np.nonzero(rref_aug_matrix[:, binary_matrix.shape[0] + _i])[0].tolist() for _i in range(vector.shape[0])]


def _single_qubit_phase_factor(pauli_ops: list[np.ndarray]) -> complex:
    """
    Compute the phase factor for a single qubit w.r.t. multiplication of Pauli operators

    Args:
        pauli_ops (list[np.ndarray]): List of Pauli operators (symplectic form) acting on the same qubit

    Returns:
        complex: Coefficient of multiplying all terms together
    """
    # Initialise as 1.0*I, then multiply with each Pauli operator acting on that qubit
    coeff, current_pauli_op = 1.0, np.zeros(2)
    for pauli_op in pauli_ops:
        # If I, just skip
        if SYMPLECTIC_VECTORS.index(current_pauli_op.tolist()) == 0:
            current_pauli_op = pauli_op
            continue
        if SYMPLECTIC_VECTORS.index(pauli_op.tolist()) == 0:
            continue
        # Multiply by some phase factor depending on what Pauli operators are involved
        coeff *= SYMPLECTIC_PHASE_TABLE[
            SYMPLECTIC_VECTORS.index(pauli_op.tolist()) - SYMPLECTIC_VECTORS.index(current_pauli_op.tolist())
        ]
        current_pauli_op = (current_pauli_op + pauli_op) % 2
    return coeff


def phase_factor(pauli_terms: list[str]) -> int:
    r"""
    Calculate the phase factor (p) in the decomposition of a Pauli string in the original Hamiltonian into a
    product of k mutually commuting Pauli terms, i.e. P_I =  p \prod_{K} \tau_k

    Args:
        pauli_terms (list[np.ndarray]): List of mutually commuting Pauli strings, given in the form of symplectic vectors.
            Each term in pauli_termsis a 1D array.

    Returns:
        int: 1 or -1
    """
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


def make_x_matrix_full_rank(stabiliser_matrix: np.ndarray) -> list[gates.Gate]:
    """
    Modifies stabiliser_matrix (in-place) to transform 'X matrix' to full rank, with H gates representing each 'swap'
    of columns between the 'Z' and 'X' matrices. Note: stabiliser_matrix should already be in reduced row echelon form

    Returns:
        list: List of H gates to be added to the circuit
    """
    gates_list = []

    dim_space = stabiliser_matrix.shape[1] // 2
    x_matrix = stabiliser_matrix[:, :dim_space]
    z_matrix = stabiliser_matrix[:, dim_space:]
    # Need to find full rank submatrix in Z matrix for each of the zero rows in the X matrix
    zero_row_indices = [_i for _i, is_zero in enumerate(np.all(x_matrix == 0, axis=1)) if is_zero]
    # Only need to do anything if there are zero rows in the X matrix
    while zero_row_indices:
        nonzero_cols_by_row = {row: list(np.nonzero(z_matrix[row, :])[0]) for row in zero_row_indices}
        # See if there are any single-element lists in the values of nonzero_cols_by_row
        no_choice_rows = [row for row, possible_cols in nonzero_cols_by_row.items() if len(possible_cols) == 1]
        chosen_row = no_choice_rows[0] if no_choice_rows else zero_row_indices[0]
        chosen_qubit = nonzero_cols_by_row[chosen_row][0]
        stabiliser_matrix[:, [chosen_qubit, chosen_qubit + dim_space]] = stabiliser_matrix[
            :, [chosen_qubit + dim_space, chosen_qubit]
        ]
        gates_list.append(gates.H(chosen_qubit))
        zero_row_indices.remove(chosen_row)
    return gates_list


def col_reduce_x_matrix(stabiliser_matrix: np.ndarray) -> list[gates.Gate]:
    """
    Modifies stabiliser_matrix in-place to transform the X matrix to I, using CNOT/SWAP gates

    Returns:
        list: List of CNOT/SWAP gates to be added to the circuit
    """
    gates_list = []
    dim, _dim_space = stabiliser_matrix.shape
    dim_space = _dim_space // 2

    # Paper used row reduction, but should be column reduction in our context
    for _i in range(dim_space):
        if _i > dim:
            break
        # Get columns with row _i != 0
        nonzero_cols = np.nonzero(stabiliser_matrix[_i, :dim_space])[0]

        # Always take the first nonzero row to sort
        _col = [_j for _j in nonzero_cols if _j >= _i][0]
        if _i not in nonzero_cols:
            stabiliser_matrix[:, [_i, _col, _i + dim_space, _col + dim_space]] = stabiliser_matrix[
                :, [_col, _i, _col + dim_space, _i + dim_space]
            ]
            gates_list.append(gates.SWAP(_i, _col))
        nonzero_cols = np.nonzero(stabiliser_matrix[_i, :dim_space])[0]
        # Remove all nonzero entries on row _i using CNOT gates
        for _col in nonzero_cols:  # Ignore first entry of nonzero_cols since effectively should be 0 now
            if _col != _i:
                # Add j^th column to i^th column
                stabiliser_matrix[:, _col] += stabiliser_matrix[:, _i]
                # Add (i+dim_space)^th column to (j+dim_space)^th column
                # RHS of stabiliser matrix should be 0 matrix, so I think can ignore...?
                stabiliser_matrix[:, dim_space + _i] += stabiliser_matrix[:, _col + dim_space]
                stabiliser_matrix %= 2
                gates_list.append(gates.CNOT(_i, _col))

    return gates_list


def zero_z_matrix(stabiliser_matrix: np.ndarray) -> list[gates.Gate]:
    """
    Modifies stabiliser_matrix in-place to transform the Z matrix to a zero matrix.
    1. S gates used to set diagonal entries on Z matrix
    2. CZ gates used to remove off-diagonal entries on Z matrix

    Returns:
        list: List of S and CZ gates to be added to the circuit
    """
    s_gates = []
    cz_gates = []
    dim, _dim_space = stabiliser_matrix.shape
    dim_space = _dim_space // 2
    # Following the algorithm in the paper, zero out the diagonal entries first
    for _i in range(dim):
        if stabiliser_matrix[_i, dim_space + _i] == 1:
            stabiliser_matrix[_i, dim_space + _i] = 0
            s_gates.append(gates.S(_i).dagger())  # Paper says S gate, but should be S.dagger?
        # Then remove the off-diagonal terms in each row
        for _j in range(dim_space):
            if _j > _i and stabiliser_matrix[_i, dim_space + _j] == 1:
                stabiliser_matrix[_i, dim_space + _j] = 0
                stabiliser_matrix[_j, dim_space + _i] = 0
                cz_gates.append(gates.CZ(_i, _j))
    return s_gates + cz_gates


def synthesise_circuit(v_basis: np.ndarray) -> list[gates.Gate]:
    """
    Build the unitary transformation circuit for rotating the initial measurement basis into the computational basis.
    The stabiliser matrix follows the format of (X|Z) matrices.

    Args:
        v_basis (np.array): Basis for the symplectic vector space of the group of commuting Pauli terms

    Returns:
        list[gates.Gate]: Gates to be added after the circuit ansatz
    """
    stabiliser_matrix = np.array(v_basis)
    n_qubits = stabiliser_matrix.shape[1] // 2
    rotation_gates = []
    # 1. Apply H gates to transform 'X matrix' to full rank
    h_gates1 = make_x_matrix_full_rank(stabiliser_matrix)
    rotation_gates += h_gates1
    # 2. Row-reduce 'X matrix' to I using CNOT/SWAP gates
    rr_gates = col_reduce_x_matrix(stabiliser_matrix)
    rotation_gates += rr_gates
    # 3. Remove all non-zero entries on 'Z matrix' using S and CZ gates
    gates_list = zero_z_matrix(stabiliser_matrix)
    rotation_gates += gates_list
    # 4. Apply H to each qubit to swap the 'X' and 'Z' matrices
    rotation_gates += [gates.H(_i) for _i in range(n_qubits)]
    return rotation_gates
