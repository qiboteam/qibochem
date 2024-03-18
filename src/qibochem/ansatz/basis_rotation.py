"""
Circuit representing an unitary rotation of the molecular (spin-)orbital basis set
"""

import numpy as np
from openfermion.linalg.givens_rotations import givens_decomposition
from qibo import gates, models
from scipy.linalg import expm

from qibochem.ansatz import ucc

# Helper functions


def unitary_rot_matrix(parameters, occ_orbitals, virt_orbitals, orbital_pairs=None, conserve_spin=False):
    r"""
    Returns the unitary rotation matrix U = exp(\kappa) mixing the occupied and virtual orbitals,
         where \kappa is an anti-Hermitian matrix

    Args:
        parameters: List/array of rotation parameters. dim = len(occ_orbitals)*len(virt_orbitals)
        occ_orbitals: Iterable of occupied orbitals
        virt_orbitals: Iterable of virtual orbitals
        orbital_pairs: (optional) Iterable of occupied-virtual orbital pairs, sorting is optional
        conserve_spin: (optional) in the case where orbital pairs are unspecified, they are generated
                        from ucc.generate_excitations for singles excitations. If True, only singles
                        excitations with spin conserved are considered
    Returns:
        exp(k): Unitary matrix of Givens rotations, obtained by matrix exponential of skew-symmetric
                kappa matrix
    """
    n_orbitals = len(occ_orbitals) + len(virt_orbitals)
    kappa = np.zeros((n_orbitals, n_orbitals))
    if orbital_pairs == None:
        # Get all possible (occ, virt) pairs
        # orbital_pairs = [(_o, _v) for _o in occ_orbitals for _v in virt_orbitals]
        orbital_pairs = ucc.generate_excitations(1, occ_orbitals, virt_orbitals, conserve_spin=conserve_spin)
    # occ/occ and virt/virt orbital mixing doesn't change the energy, so can ignore
    for _i, (_occ, _virt) in enumerate(orbital_pairs):
        kappa[_occ, _virt] = parameters[_i]
        kappa[_virt, _occ] = -parameters[_i]
    return expm(kappa)


def swap_matrices(permutations, n_qubits):
    r"""
    Build matrices that permute (swap) the columns of a given matrix

    Args:
        permutations [List(tuple, ), ]: List of lists of 2-tuples permuting two consecutive numbers
            e.g. [[(0, 1), (2, 3)], [(1, 2)], ...]
        n_qubits: Dimension of matrix (\exp(\kappa) matrix) to be operated on i.e. # of qubits

    Returns:
        List of matrices that carry out \exp(swap) for each pair in permutations
    """
    exp_swaps = []
    _temp = np.array([[-1.0, 1.0], [1.0, -1.0]])
    for pairs in permutations:
        gen = np.zeros((n_qubits, n_qubits))
        for pair in pairs:
            # Add zeros to pad out the (2, 2) matrix to (n_qubits, n_qubits) with 0.
            gen += np.pad(_temp, ((pair[0], n_qubits - pair[1] - 1),), "constant")
        # Take matrix exponential, remove all entries close to zero, and convert to real matrix
        exp_mat = expm(-0.5j * np.pi * gen)
        exp_mat[np.abs(exp_mat) < 1e-10] = 0.0
        exp_mat = np.real_if_close(exp_mat)
        # Add exp_mat to exp_swaps once cleaned up
        exp_swaps.append(exp_mat)
    return exp_swaps


def givens_rotation_parameters(n_qubits, exp_k, n_occ):
    r"""
    Get the parameters of the Givens rotation matrix obtained after decomposing \exp(\kappa)

    Args:
        n_qubits: Number of qubits (i.e. spin-orbitals)
        exp_k: Unitary rotation matrix
        n_occ: Number of occupied orbitals
    """
    # List of 2-tuples to link all qubits via swapping
    swap_pairs = [
        [(_i, _i + 1) for _i in range(_d % 2, n_qubits - 1, 2)]
        # Only 2 possible ways of swapping: [(0, 1), ... ] or [(1, 2), ...]
        for _d in range(2)
    ]
    # Get their corresponding matrices
    swap_unitaries = swap_matrices(swap_pairs, n_qubits)

    # Create a copy of exp_k for rotating
    exp_k_shift = np.copy(exp_k)
    # Apply the column-swapping transformations
    for u_rot in swap_unitaries:
        exp_k_shift = u_rot @ exp_k_shift
    # Only want the n_occ by n_qubits (n_orb) sub-matrix
    qmatrix = exp_k_shift.T[:n_occ, :]

    # Apply Givens decomposition of the submatrix to get a list of individual Givens rotations
    # (orb1, orb2, theta, phi), where orb1 and orb2 are rotated by theta(real) and phi(imag)
    qdecomp, _, _ = givens_decomposition(qmatrix)
    # Lastly, reverse the list of Givens rotations (because U = G_k ... G_2 G_1)
    return list(reversed(qdecomp))


def givens_rotation_gate(n_qubits, orb1, orb2, theta):
    r"""
    Circuit corresponding to a Givens rotation between two qubits (spin-orbitals)

    Args:
        n_qubits: Number of qubits in the quantum circuit
        orb1, orb2: Orbitals used for the Givens rotation
        theta: Rotation parameter

    Returns:
        circuit: Qibo Circuit object representing a Givens rotation between orb1 and orb2
    """
    # Define 2x2 sqrt(iSwap) matrix
    # iswap_mat = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    # Build and add the gates to circuit
    circuit = models.Circuit(n_qubits)
    # circuit.add(gates.GeneralizedfSim(orb1, orb2, iswap_mat, 0.0, trainable=False))
    circuit.add(gates.SiSWAP(orb1, orb2))
    circuit.add(gates.RZ(orb1, -theta))
    circuit.add(gates.RZ(orb2, theta + np.pi))
    # circuit.add(gates.GeneralizedfSim(orb1, orb2, iswap_mat, 0.0, trainable=False))
    circuit.add(gates.SiSWAP(orb1, orb2))
    circuit.add(gates.RZ(orb1, np.pi, trainable=False))
    return circuit


def br_circuit(n_qubits, parameters, n_occ):
    r"""
    Google's basis rotation circuit, applied between the occupied/virtual orbitals. Forms the exp(kappa) matrix, decomposes
    it into Givens rotations, and sets the circuit parameters based on the Givens rotation decomposition. Note: Supposed
    to be used with the JW fermion-to-qubit mapping

    Args:
        n_qubits: Number of qubits in the quantum circuit
        parameters: Rotation parameters for exp(kappa); Must have (n_occ * n_virt) parameters
        n_occ: Number of occupied orbitals

    Returns:
        Qibo ``Circuit``: Circuit corresponding to the basis rotation ansatz between the occupied and virtual orbitals
    """
    assert len(parameters) == (n_occ * (n_qubits - n_occ)), "Need len(parameters) == (n_occ * n_virt)"
    # Unitary rotation matrix \exp(\kappa)
    exp_k = unitary_rot_matrix(parameters, range(n_occ), range(n_occ, n_qubits))
    # List of Givens rotation parameters
    g_rotation_parameters = givens_rotation_parameters(n_qubits, exp_k, n_occ)

    # Start building the circuit
    circuit = models.Circuit(n_qubits)
    # Add the Givens rotation gates
    for g_rot_parameter in g_rotation_parameters:
        for orb1, orb2, theta, _phi in g_rot_parameter:
            assert np.allclose(_phi, 0.0), "Unitary rotation is not real!"
            circuit += givens_rotation_gate(n_qubits, orb1, orb2, theta)
    return circuit


# clements qr routines


def givens_qr_decompose(U):
    r"""
    Clements scheme QR decompose a unitary matrix U using Givens rotations
    see arxiv:1603.08788
    Args:
        U: unitary rotation matrix
    Returns:
        z_angles: array of rotation angles
        U: final unitary after QR decomposition, should be an identity
    """

    def move_step(U, row, col, row_change, col_change):
        """
        internal function to move a step in Givens QR decomposition of
        unitary rotation matrix
        """
        row += row_change
        col += col_change
        return U, row, col

    def row_op(U, row, col):
        """
        internal function to zero out a row using Givens rotation
        with angle z
        """
        Uc = U.copy()
        srow = row - 1
        scol = col
        z = np.arctan2(-Uc[row][col], Uc[srow][scol])
        temp_srow = Uc[srow, :]
        temp_row = Uc[row, :]
        new_srow = np.cos(z) * temp_srow - np.sin(z) * temp_row
        new_row = np.sin(z) * temp_srow + np.cos(z) * temp_row
        Uc[srow, :] = new_srow
        Uc[row, :] = new_row
        return z, Uc

    def col_op(U, row, col):
        """
        internal function to zero out a column using Givens rotation
        with angle z
        """
        Uc = U.copy()
        srow = row
        scol = col + 1
        z = np.arctan2(-Uc[row][col], Uc[srow][scol])
        temp_scol = Uc[:, scol]
        temp_col = Uc[:, col]
        new_scol = np.cos(z) * temp_scol - np.sin(z) * temp_col
        new_col = np.sin(z) * temp_scol + np.cos(z) * temp_col
        Uc[:, scol] = new_scol
        Uc[:, col] = new_col
        return z, Uc

    N = len(U[0])

    z_array = []

    # start QR from bottom left element
    row = N - 1
    col = 0
    z, U = col_op(U, row, col)
    z_array.append(z)
    # traverse the U matrix in diagonal-zig-zag manner until the main
    # diagonal is reached
    # if move = up, do a row op
    # if move = diagonal-down-right, do a row op
    # if move = right, do a column op
    # if move = diagonal-up-left, do a column op
    while row != 1:
        U, row, col = move_step(U, row, col, -1, 0)
        z, U = row_op(U, row, col)
        z_array.append(z)
        while row < N - 1:
            U, row, col = move_step(U, row, col, 1, 1)
            z, U = row_op(U, row, col)
            z_array.append(z)
        if col != N - 2:
            U, row, col = move_step(U, row, col, 0, 1)
            z, U = col_op(U, row, col)
            z_array.append(z)
        else:
            break
        while col > 0:
            U, row, col = move_step(U, row, col, -1, -1)
            z, U = col_op(U, row, col)
            z_array.append(z)

    return z_array, U


def basis_rotation_layout(N):
    r"""
    generates the layout of the basis rotation circuit for Clements scheme QR decomposition
    Args:
        N: number of qubits/modes
    Returns:
        A:  NxN matrix, with -1 being null, 0 is the control and integers
            1 or greater being the index for angles in clements QR decomposition of
            the unitary matrix representing the unitary transforms that
            rotate the basis
    """

    def move_step(A, row, col, row_change, col_change):
        """
        internal function to move a step in gate layout of br circuit
        """
        row += row_change
        col += col_change
        return A, row, col, updown

    def check_top(A, row, col, row_change, col_change):
        """
        check if we are at the top of layout matrix,
        return false if we are at the top, i.e. row == 1
        (row 0 is not assigned any operation; just control)
        """
        move_step(A, row, col, row_change, col_change)
        return row > 1

    def check_bottom(A, row, col, row_change, col_change):
        """
        check if we are at the bottom of layout matrix
        return false if we are at the bottom, i.e. row == N-1
        """
        N = len(A)
        move_step(A, row, col, row_change, col_change)
        return row < N - 1

    def jump_right(A, row, col, updown):
        """ """
        N = len(A)
        row = N - 2 - col
        col = N - 1
        updown = -1
        return A, row, col, updown

    def jump_left(A, row, col, updown):
        N = len(A)
        row = N + 1 - col
        col = 0
        updown = 1
        return A, row, col, updown

    def assign_element(A, row, col, k):
        A[row - 1][col] = 0
        A[row][col] = k
        return A, row, col

    nangles = (N - 1) * N // 2  # half-triangle
    A = np.full([N, N], -1, dtype=int)

    # first step
    row = 1
    col = 0
    k = 1
    A, row, col = assign_element(A, row, col, k)
    k += 1
    updown = 1

    while k <= nangles:

        if updown == 1:
            if check_top(A, row, col, -1, 1):
                A, row, col, updown = move_step(A, row, col, -1, 1)
                A, row, col = assign_element(A, row, col, k)
            else:  # jump
                # print('jump_right')
                A, row, col, updown = jump_right(A, row, col, updown)
                A, row, col = assign_element(A, row, col, k)
        elif updown == -1:
            if check_bottom(A, row, col, 1, -1):
                A, row, col, updown = move_step(A, row, col, 1, -1)
                A, row, col = assign_element(A, row, col, k)
            else:  # jump
                # print('jump left')
                A, row, col, updown = jump_left(A, row, col, updown)
                A, row, col = assign_element(A, row, col, k)
        else:
            raise ValueError("Bad direction")

        # print(row, col)
        k += 1

    return A


def basis_rotation_gates(A, z_array, parameters):
    r"""
    places the basis rotation gates on circuit in the order of Clements scheme QR decomposition
    Args:
        A:
            NxN matrix, with -1 being null, 0 is the control and integers
            1 or greater being the index for angles in clements QR decomposition of
            the unitary matrix representing the unitary transforms that
            rotate the basis
        z_array:
            array of givens rotation angles in order of traversal from
            QR decomposition
        parameters:
            array of parameters in order of traversal from QR decomposition
    Outputs:
        gate_list:
            list of gates which implement the basis rotation using Clements scheme QR decomposition
    """
    N = len(A[0])
    gate_list = []

    #
    for j in range(N):
        for i in range(N):
            if A[i][j] == 0:
                print("CRY", i, i + 1, j, z_array[A[i + 1][j] - 1])
                gate_list.append(gates.CNOT(i + 1, i))
                gate_list.append(gates.CRY(i, i + 1, z_array[A[i + 1][j] - 1]))
                gate_list.append(gates.CNOT(i + 1, i))

    return gate_list
