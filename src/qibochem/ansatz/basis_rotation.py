"""
Circuit representing an unitary rotation of the molecular (spin-)orbital basis set
"""

import numpy as np
from qibo import gates, models
from scipy.linalg import expm

from qibochem.ansatz import ucc
from qibochem.driver import hamiltonian

# Helper functions


def unitary(occ_orbitals, virt_orbitals, parameters=None):
    r"""
    Returns the unitary rotation matrix :math: `U = \exp(\kappa)` mixing the occupied and virtual orbitals.
    Orbitals are arranged in alternating spins, e.g. for 4 occupied orbitals [0,1,2,3], the spins are arranged as [0a, 0b, 1a, 1b]
    Dimension for the array of rotation parameters is len(occ_orbitals_a)*len(virt_orbitals_a) + len(occ_orbitals_b)*len(virt_orbitals_b)
    The current implementation of this function only accommodates systems with all electrons paired, number of alpha and beta spin electrons are equal
    Args:
        occ_orbitals: Iterable of occupied orbitals
        virt_orbitals: Iterable of virtual orbitals
        parameters: List/array of rotation parameters
            dimension = len(occ_orbitals)*len(virt_orbitals)
    Returns:
        exp(k): Unitary matrix of Givens rotations, obtained by matrix exponential of skew-symmetric
                kappa matrix
    """

    # conserve_spin has to be true for SCF/basis_rotation cases, else expm(k) is not unitary
    ov_pairs = ucc.generate_excitations(1, occ_orbitals, virt_orbitals, conserve_spin=True)
    # print('ov_pairs presort', ov_pairs)
    ov_pairs = ucc.sort_excitations(ov_pairs)
    # print('ov_pairs sorted ', ov_pairs)
    n_theta = len(ov_pairs)
    if parameters is None:
        print("basis rotation: parameters not specified")
        print("basis rotation: using default occ-virt rotation parameter value = 0.0")
        parameters = np.zeros(n_theta)
    elif isinstance(parameters, float):
        print("basis rotation: using uniform value of", parameters, "for each parameter value")
        parameters = np.full(n_theta, parameters)
    else:
        if len(parameters) != n_theta:
            raise IndexError("parameter array specified has bad size or type")
        print("basis rotation: loading parameters from input")

    n_orbitals = len(occ_orbitals) + len(virt_orbitals)
    kappa = np.zeros((n_orbitals, n_orbitals))

    for _i, (_occ, _virt) in enumerate(ov_pairs):
        kappa[_occ, _virt] = parameters[_i]
        kappa[_virt, _occ] = -parameters[_i]

    return expm(kappa), parameters


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
        # else:
        #    break
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
        # else:
        #    raise ValueError("Bad direction")

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
        ordered_angles:
            list of angles ordered by sequence of singles excitation gates added to circuit

    """
    N = len(A[0])
    gate_list = []
    ordered_angles = []
    #
    for j in range(N):
        for i in range(N):
            if A[i][j] == 0:
                # gate_list.append(gates.CNOT(i + 1, i))
                # gate_list.append(gates.CRY(i, i + 1, z_array[A[i + 1][j] - 1]))
                gate_list.append(gates.GIVENS(i + 1, i, z_array[A[i + 1][j] - 1]))
                ordered_angles.append(z_array[A[i + 1][j] - 1])
                # gate_list.append(gates.CNOT(i + 1, i))

    return gate_list, ordered_angles
