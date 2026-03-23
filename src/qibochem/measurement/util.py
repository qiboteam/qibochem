"""
Utility functions for optimising measurements and calculation of expectation value
"""

import networkx as nx
import numpy as np
from qibo import gates

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


def binary_gaussian_elimination(vector_space):
    """
    Carries out Gaussian elimination on a binary (!) vector_space to obtain a basis for vector_space.
    Reduces vector_space to its (unique) reduced row echelon form, and removes any zero rows as well

    Args:
        vector_space (np.array, dtype=int): Binary vector space

    Returns:
        np.array: Basis set for vector_space
    """
    cp_vector_space = np.array(vector_space)
    # Get a list of non-zero columns
    # nonzero_cols = np.nonzero(np.any(cp_vector_space, axis=0))[0]

    dim = vector_space.shape[0]
    # Swap the rows in the vector space to get its row echelon form
    for _i in range(dim):
        subspace_to_sort = cp_vector_space[_i:, :]
        if not np.any(subspace_to_sort):
            break

        nonzero_cols = np.nonzero(np.any(subspace_to_sort, axis=0))[0]
        # Always take the first nonzero column to sort
        _col = nonzero_cols[0]

        # col_indices = cp_vector_space[_i:, _col].argsort()[::-1]
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


def binary_nullspace(binary_matrix):
    """
    Finds the nullspace of a binary_matrix, i.e. Ax = 0

    Returns:
        np.array: Nullspace of binary matrix
    """
    dim = binary_matrix.shape[0]
    # Form the augmented matrix
    aug_matrix = np.concatenate((binary_matrix.T, np.identity(binary_matrix.shape[1])), axis=1)
    rref_aug_matrix = binary_gaussian_elimination(aug_matrix)
    nullspace = rref_aug_matrix[dim:, dim:]
    return nullspace.astype(int)


def langrangian_subspace(vector_space):
    """
    Find the Lagrangian subspace of some vector space. (The symplectic nullspace in this context)

    Returns:
        np.array: Basis vectors of vector_space. Shape of the array should be (N, 2N)
    """
    cp_vector_space = np.array(vector_space)
    # While loop to remove rows from cp_vector_space until cp_vector_space.shape matches (N, 2N)
    while True:
        # anticommuting_vector_dict = {}
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


def sort_tau_terms(v_basis):
    """
    Sorts the individual vectors in v_basis s.t. the i'th term of basis vector i is NOT I, e.g.
    [['X0', 'X2'], ['Z1', 'X3', 'Z4', 'X5'], ['Z0', 'Z2'], ['Z1'], ['Z3', 'Z5'], ['Z4']]
    will return
    [['X0', 'X2'], ['Z1'], ['Z0', 'Z2'], ['Z3', 'Z5'], ['Z4'], ['Z1', 'X3', 'Z4', 'X5']]

    Returns:
        list: Sorted list of basis vectors that can be used for finding sigma_i directly
    """
    # Convert the basis set to strings for easier sorting
    pauli_terms = [symplectic_to_pauli(vector) for vector in v_basis]
    dim = len(pauli_terms)

    sorted_terms = {}
    while True:
        possible_terms = {
            _i: [term for term in pauli_terms if any(int(_op[1:]) == _i for _op in term)]
            for _i in range(dim)
            if _i not in sorted_terms
        }
        # Remove terms that only have a single possibility
        single_choices = [_i for _i, terms in possible_terms.items() if len(terms) == 1]
        if single_choices:
            # Should have no more overlapping terms? TODO: If got then how?
            for qubit in single_choices:
                sorted_terms[qubit] = pauli_terms.pop(pauli_terms.index(possible_terms[qubit][0]))
        # Select based on the first remaining unassigned qubit
        else:
            least_choices = min(len(terms) for terms in possible_terms.values())
            n_choices = [_q for _q, terms in possible_terms.items() if len(terms) == least_choices]
            qubit = min(n_choices)
            term_to_remove = min(possible_terms[qubit], key=len)
            sorted_terms[qubit] = pauli_terms.pop(pauli_terms.index(term_to_remove))
        if not pauli_terms:
            break
    # Convert the strings back to symplectic vectors and return the whole array
    return np.array([pauli_to_symplectic(sorted_terms[_i], dim) for _i in range(dim)])


def get_sigma_terms(tau_terms):
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
                # symplectic_inner_product(new_tau_terms[_j], sigma_i)*tau_i if _j > _i else np.zeros(2*dim)
                for _j in range(dim)
            ]
        ).astype(int)
        new_tau_terms = new_tau_terms % 2

    return new_tau_terms, np.array(sigma_terms)


def solve_linear_system_single_vector(A, b):
    """
    Solve the (binary) linear system Ax = b, where b is a single row vector

    Returns:
        list: Each item in the list corresponds to the respective vectors in b.
    """
    # Form the augmented matrix
    aug_matrix = np.concatenate((A.T, b[:, None]), axis=1)
    rref_aug_matrix = binary_gaussian_elimination(aug_matrix)
    # Find the non-zero entries in each column on the RHS of rref_aug_matrix, which is the solution for respective vector in b
    return np.nonzero(rref_aug_matrix[:, A.shape[0]])[0].tolist()


def solve_linear_system(A, b):
    """
    Solve the (binary) linear system Ax = b

    Returns:
        list: Each item in the list corresponds to the respective vectors in b.
    """
    # Form the augmented matrix
    aug_matrix = np.concatenate((A, b), axis=0).T
    rref_aug_matrix = binary_gaussian_elimination(aug_matrix)
    # Find the non-zero entries in each column on the RHS of rref_aug_matrix, which is the solution for respective vector in b
    return [np.nonzero(rref_aug_matrix[:, A.shape[0] + _i])[0].tolist() for _i in range(b.shape[0])]


def phase_factor(tau_k_terms):
    r"""
    Calculate the phase factor (p) in the decomposition of a Pauli string in the original Hamiltonian into a
    product of k mutually commuting Pauli terms, i.e. P_I =  p \prod_{K} \tau_k

    Args:
        tau_k_terms: Iterable of mutually commuting Pauli products, given in the form of symplectic vectors.
            Note: Each tau_k term is a 1D array.

    Returns:
        int: 1 or -1
    """
    dim = tau_k_terms[0].shape[0] // 2
    # Singleton case is trivial: 1
    if len(tau_k_terms) == 1:
        return 1
    # >1 terms: Need to take the product of every Pauli operator of every term in tau_k_terms
    phases = (1.0, 1.0j, -1.0j)
    symplectic_vectors = [[0, 0], [1, 0], [1, 1], [0, 1]]  # I, X, Y, Z

    # Split up the vectors by imdividual qubits, then take the product for each qubit
    split_vectors_by_qubit = [[tau_k[[_i, _i + dim]] for _i in range(dim)] for tau_k in tau_k_terms]

    coefficient = 1.0
    for qubit in range(dim):
        # Get all Pauli operators for a particular qubit
        pauli_ops = [vector[qubit] for vector in split_vectors_by_qubit]
        # Initialise as 1.0*I, then multiply with each Pauli operator acting on that qubit
        coeff, current_pauli_op = 1.0, np.zeros(2)
        for pauli_op in pauli_ops:
            # To handle I cases
            if symplectic_vectors.index(current_pauli_op.tolist()) == 0:
                current_pauli_op = pauli_op
                continue
            if symplectic_vectors.index(pauli_op.tolist()) == 0:
                continue
            # Multiply by some phase factor depending on what Pauli operators are involved
            coeff *= phases[
                symplectic_vectors.index(pauli_op.tolist()) - symplectic_vectors.index(current_pauli_op.tolist())
            ]
            current_pauli_op = (current_pauli_op + pauli_op) % 2
        coefficient *= coeff
    return int(np.real_if_close(coefficient))


def make_x_matrix_full_rank(stabliser_matrix):
    """
    Modifies stabliser_matrix in-place to transform 'X matrix' to full rank, with H gates representing each 'swap' of columns between the 'Z' and 'X' matrices.
    stabliser_matrix should already be in reduced row echelon form

    Returns:
        list: List of H gates to be added to the circuit
    """
    gates_list = []
    _dim, _dim_space = stabliser_matrix.shape
    dim_space = _dim_space // 2

    # TODO: Try to work directly in the stabliser_matrix instead of defining x/z_matrix
    x_matrix = stabliser_matrix[:, :dim_space]
    z_matrix = stabliser_matrix[:, dim_space:]

    # Need to find full rank submatrix in Z matrix for each of the zero rows in the X matrix
    zero_row_indices = [_i for _i, is_zero in enumerate(np.all(x_matrix == 0, axis=1)) if is_zero]
    # Only need to do anything if there are zero rows in the X matrix
    while zero_row_indices:
        nonzero_cols_by_row = {row: list(np.nonzero(z_matrix[row, :])[0]) for row in zero_row_indices}
        # See if there are any single-element lists in the values of nonzero_cols_by_row
        no_choice_rows = [row for row, possible_cols in nonzero_cols_by_row.items() if len(possible_cols) == 1]
        chosen_row = no_choice_rows[0] if no_choice_rows else zero_row_indices[0]
        chosen_qubit = nonzero_cols_by_row[chosen_row][0]
        stabliser_matrix[:, [chosen_qubit, chosen_qubit + dim_space]] = stabliser_matrix[
            :, [chosen_qubit + dim_space, chosen_qubit]
        ]
        gates_list.append(gates.H(chosen_qubit))
        zero_row_indices.remove(chosen_row)
    return gates_list


def col_reduce_x_matrix(stabliser_matrix):
    """
    Modifies stabliser_matrix in-place to transform the X matrix to I, using CNOT/SWAP gates

    Returns:
        list: List of CNOT/SWAP gates to be added to the circuit
    """
    gates_list = []
    dim, _dim_space = stabliser_matrix.shape
    dim_space = _dim_space // 2

    # Paper used row reduction, but should be column reduction in our context
    for _i in range(dim_space):
        if _i > dim:
            break
        # Get columns with row _i != 0
        nonzero_cols = np.nonzero(stabliser_matrix[_i, :dim_space])[0]

        # Always take the first nonzero row to sort
        _col = [_j for _j in nonzero_cols if _j >= _i][0]
        if _i not in nonzero_cols:
            stabliser_matrix[:, [_i, _col, _i + dim_space, _col + dim_space]] = stabliser_matrix[
                :, [_col, _i, _col + dim_space, _i + dim_space]
            ]
            gates_list.append(gates.SWAP(_i, _col))
        nonzero_cols = np.nonzero(stabliser_matrix[_i, :dim_space])[0]
        # Remove all nonzero entries on row _i using CNOT gates
        for _col in nonzero_cols:  # Ignore first entry of nonzero_cols since effectively should be 0 now
            if _col != _i:
                # Add j^th column to i^th column
                stabliser_matrix[:, _col] += stabliser_matrix[:, _i]
                # Add (i+dim_space)^th column to (j+dim_space)^th column
                # But RHS of stabiliser matrix should be 0 matrix, so I think can ignore...?
                stabliser_matrix[:, dim_space + _i] += stabliser_matrix[:, _col + dim_space]
                # Mod 2
                stabliser_matrix %= 2
                gates_list.append(gates.CNOT(_i, _col))

    return gates_list


def zero_z_matrix(stabliser_matrix):
    """
    Modifies stabliser_matrix in-place to transform the Z matrix to a zero matrix.
    1. S gates used to set diagonal entries on Z matrix
    2. CZ gates used to remove off-diagonal entries on Z matrix

    Returns:
        list: List of S and CZ gates to be added to the circuit
    """
    s_gates = []
    cz_gates = []
    dim, _dim_space = stabliser_matrix.shape
    dim_space = _dim_space // 2
    # Following the algorithm in the paper, zero out the diagonal entries first
    for _i in range(dim):
        if stabliser_matrix[_i, dim_space + _i] == 1:
            s_gates.append(gates.S(_i).dagger())  # Paper says S gate, but should be S.dagger?
            stabliser_matrix[_i, dim_space + _i] = 0
        # Then remove the off-diagonal terms in each row
        for _j in range(dim_space):
            if _j > _i and stabliser_matrix[_i, dim_space + _j] == 1:
                cz_gates.append(gates.CZ(_i, _j))
                stabliser_matrix[_i, dim_space + _j] = 0
                stabliser_matrix[_j, dim_space + _i] = 0
    return s_gates + cz_gates


def synthesise_circuit(v_basis):
    """
    Build the unitary transformation circuit according to the algorithm

    Args:
        v_basis (np.array): Basis for the symplectic vector space of the group of commuting Pauli terms

    Returns:
        list: List of gates to be applied after the circuit ansatz for rotating the initial measurement basis into the computational basis
    """
    stabliser_matrix = np.array(v_basis)
    n_qubits = stabliser_matrix.shape[1] // 2
    rotation_gates = []
    # 1. Apply H gates to transform 'X matrix' to full rank
    h_gates1 = make_x_matrix_full_rank(stabliser_matrix)
    rotation_gates += h_gates1
    # 2. Row-reduce 'X matrix' to I using CNOT/SWAP gates
    rr_gates = col_reduce_x_matrix(stabliser_matrix)
    rotation_gates += rr_gates
    # 3. Remove all non-zero entries on 'Z matrix' using S and CZ gates
    gates_list = zero_z_matrix(stabliser_matrix)
    rotation_gates += gates_list
    # 4. Apply H to each qubit to swap the 'X' and 'Z' matrices
    rotation_gates += [gates.H(_i) for _i in range(n_qubits)]
    # stabliser_matrix[:, list(range(stabliser_matrix.shape[1]))] = (
    #     stabliser_matrix[:, list(range(n_qubits, 2*n_qubits)) + list(range(n_qubits))]
    # )
    return rotation_gates
