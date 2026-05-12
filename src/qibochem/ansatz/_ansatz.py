"""Helper functions for the `ansatz` module"""

from collections.abc import Iterable, Sequence

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error
from scipy.linalg import expm

from qibochem.ansatz.utils import generate_excitations, sort_excitations


def _bk_matrix_power2(dims: int) -> np.ndarray:
    """Build the Bravyi-Kitaev matrix of dimension ``dims`` :math:`d = 2^{n}` recursively"""
    # Base case
    if dims == 1:
        return np.ones((1, 1), dtype=np.int8)

    # Recursive definition
    smaller_bk_matrix = _bk_matrix_power2(dims - 1)
    top_right = np.zeros((2 ** (dims - 2), 2 ** (dims - 2)), dtype=np.int8)
    top_half = np.concatenate((smaller_bk_matrix, top_right), axis=1)

    bottom_left = np.concatenate(
        (
            np.zeros(((2 ** (dims - 2)) - 1, 2 ** (dims - 2)), dtype=np.int8),
            np.ones((1, 2 ** (dims - 2)), dtype=np.int8),
        ),
        axis=0,
    )
    bottom_half = np.concatenate((bottom_left, smaller_bk_matrix), axis=1)

    # Combine top and bottom half
    return np.concatenate((top_half, bottom_half), axis=0)


def _bk_matrix(dims: int) -> np.ndarray:
    """Exact Brayvi-Kitaev matrix of size dims, obtained by slicing a larger BK matrix with dimension 2**m > n

    TODO: Update the occupation number vector using the update, parity, and flip set instead?
        Not sure if necessary; i.e. size of BK matrix probably not comparable to the memory needed
        for a classical simulation?

    Args:
        dims (int): Size of BK matrix
    """
    # Build bk_matrix_power2(m), where 2**m > dims
    min_bk_size = int(np.ceil(np.log2(dims))) + 1
    min_bk_matrix = _bk_matrix_power2(min_bk_size)
    # Then use array slicing to get the actual BK matrix
    return min_bk_matrix[:dims, :dims]


def _expi_pauli(n_qubits: int, pauli_string: str, theta: float, **kwargs):
    """
    Build circuit representing exp(i*theta*pauli_string)

    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        pauli_string (str): Single Pauli term in the format: ``"X0 Y3 Z11 X1"``
        theta (float): Rotation parameter
        kwargs (dict, optional): Qibo Circuit keyword arguments

    Returns:
        Circuit: Circuit representing exp(i*theta*pauli_string)
    """
    # Split pauli_string into an ordered list, e.g. [(0, "X"), ..., (11, "Z")]
    pauli_ops = sorted(((int(_op[1:]), _op[0]) for _op in pauli_string.split()), key=lambda x: x[0])
    n_pauli_ops = len(pauli_ops)

    # Generate the list of basis change gates using the pauli_ops list. "X": H, "Y": S.dagger and H
    basis_changes = []
    for qubit, pauli_op in pauli_ops:
        if pauli_op == "Y":
            basis_changes.append(gates.S(qubit).dagger())
        if pauli_op not in ("I", "Z"):
            basis_changes.append(gates.H(qubit))

    # Build the circuit
    circuit = Circuit(n_qubits, **kwargs)
    # 1. Change to X/Y where necessary
    circuit.add(basis_changes)
    # 2. Add CNOTs to all pairs of qubits in pauli_ops, starting from the last letter
    circuit.add(gates.CNOT(pauli_ops[_i][0], pauli_ops[_i - 1][0]) for _i in range(n_pauli_ops - 1, 0, -1))
    # 3. Add RZ gate to last element of pauli_ops
    circuit.add(gates.RZ(pauli_ops[0][0], -2.0 * theta))  # -2.0 coefficient needed for applying with a RZ gate
    # 4. Add CNOTs to all pairs of qubits in pauli_ops
    circuit.add(gates.CNOT(pauli_ops[_i + 1][0], pauli_ops[_i][0]) for _i in range(n_pauli_ops - 1))
    # 3. Change back to the Z basis
    circuit.add(_gate.dagger() for _gate in reversed(basis_changes))
    return circuit


def _basis_rotation_unitary(
    occ_orbitals: Iterable[int], virt_orbitals: Iterable[int], parameters: Iterable[float] | float
) -> np.ndarray:
    r"""
    Constructs the unitary rotation matrix :math:`U = \exp(\kappa)` mixing the occupied and virtual orbitals. Orbitals
    are arranged in alternating spins, e.g. for 4 occupied orbitals [0,1,2,3], the spins are arranged as
    [0a, 0b, 1a, 1b]. The current implementation of this function only accommodates systems with all electrons paired,
    and an equal number of alpha and beta spin electrons.

    Args:
        occ_orbitals (Iterable[int]): Occupied orbitals
        virt_orbitals (Iterable[int]): Virtual orbitals
        parameters (Iterable[float] | float): Rotation parameters; must have `len(occ_orbitals)*len(virt_orbitals)` elements

    Returns:
        np.ndarray: Unitary matrix of Givens rotations, obtained by matrix exponential of skew-symmetric kappa matrix
    """
    # conserve_spin has to be true for SCF/basis_rotation cases, else expm(k) is not unitary
    ov_pairs = sort_excitations(generate_excitations(1, occ_orbitals, virt_orbitals, conserve_spin=True))
    n_theta = len(ov_pairs)
    if parameters is None:
        parameters = np.zeros(n_theta)
    elif isinstance(parameters, float):
        parameters = np.full(n_theta, parameters)
    else:
        if len(parameters) != n_theta:
            raise_error(IndexError, "parameter argument specified has bad size or type")

    n_orbitals = len(occ_orbitals) + len(virt_orbitals)
    kappa = np.zeros((n_orbitals, n_orbitals))

    for _i, (_occ, _virt) in enumerate(ov_pairs):
        kappa[_occ, _virt] = parameters[_i]
        kappa[_virt, _occ] = -parameters[_i]

    return expm(kappa)


def _qr_decompose_givens(unitary_matrix: np.ndarray) -> np.ndarray:
    r"""
    Clements scheme to QR decompose a unitary matrix using Givens rotations. See arxiv:1603.08788

    Args:
        unitary_matrix (np.ndarray): Unitary rotation matrix

    Returns:
        list[float]: Rotation angles
    """

    def row_op(unitary_matrix, row, col):
        """Zero out a row using Givens rotation; returns the rotation angle"""
        srow = row - 1
        angle = np.arctan2(-unitary_matrix[row][col], unitary_matrix[srow][col])
        new_srow = np.cos(angle) * unitary_matrix[srow, :] - np.sin(angle) * unitary_matrix[row, :]
        new_row = np.sin(angle) * unitary_matrix[srow, :] + np.cos(angle) * unitary_matrix[row, :]
        unitary_matrix[srow, :] = new_srow
        unitary_matrix[row, :] = new_row
        return angle

    def col_op(unitary_matrix, row, col):
        """Zero out a column using Givens rotation; returns the rotation angle"""
        scol = col + 1
        angle = np.arctan2(-unitary_matrix[row][col], unitary_matrix[row][scol])
        new_scol = np.cos(angle) * unitary_matrix[:, scol] - np.sin(angle) * unitary_matrix[:, col]
        new_col = np.sin(angle) * unitary_matrix[:, scol] + np.cos(angle) * unitary_matrix[:, col]
        unitary_matrix[:, scol] = new_scol
        unitary_matrix[:, col] = new_col
        return angle

    dim = unitary_matrix.shape[0]

    z_angles = []
    # Start QR from bottom left element
    row, col = (dim - 1, 0)
    z_angles.append(col_op(unitary_matrix, row, col))
    # Traverse the unitary_matrix in diagonal-zig-zag manner until the main diagonal is reached
    # if move = up, do a row op
    # if move = diagonal-down-right, do a row op
    # if move = right, do a column op
    # if move = diagonal-up-left, do a column op
    while row != 1:
        row += -1
        z_angles.append(row_op(unitary_matrix, row, col))
        while row < dim - 1:
            row += 1
            col += 1
            z_angles.append(row_op(unitary_matrix, row, col))
        if col != dim - 2:
            col += 1
            z_angles.append(col_op(unitary_matrix, row, col))
        while col > 0:
            row += -1
            col += -1
            z_angles.append(col_op(unitary_matrix, row, col))
    return z_angles


def _basis_rotation_layout(nqubits: int, z_angles: Sequence[float]) -> list[tuple[int, int, float]]:
    """
    Get qubit indices and rotation angles for the Givens gates to be added to construct the basis rotation circuit

    Args:
        nqubits (int): Number of qubits/modes
        z_angles (Sequence[float]): Rotation angles

    Returns:
        list[tuple[int, int, float]]: Qubits and rotation angles of Givens gates to be added
    """

    def assign_element(array, row, col, k):
        """Helper function to set matrix elements of array"""
        array[row - 1][col] = 0
        array[row][col] = k

    array = np.full((nqubits, nqubits), -1, dtype=int)
    # First step
    row, col = (1, 0)
    k = 1
    assign_element(array, row, col, k)
    k += 1
    updown = 1
    while k <= ((nqubits - 1) * nqubits // 2):  # Half-triangle
        if updown == 1:
            # Check if reached top of layout matrix, i.e. row == 1. (row 0 not assigned any operation; just control)
            if row > 1:
                row += -1
                col += 1
                assign_element(array, row, col, k)
            else:
                # Jump right
                row = nqubits - 2 - col
                col = nqubits - 1
                updown = -1
                assign_element(array, row, col, k)
        elif updown == -1:
            # Check if at bottom of layout matrix
            if row < nqubits - 1:
                row += 1
                col += -1
                assign_element(array, row, col, k)
            else:
                # Jump left
                row = nqubits + 1 - col
                col = 0
                updown = 1
                assign_element(array, row, col, k)
        k += 1
    # Collate the zero indices of array
    zero_indices = np.where(array == 0)  # 2-tuple of 1D arrays
    # Unpack the indices into 2-tuples (sorted by column), and add the corresponding rotation angles from z_angles
    result = sorted(
        ((int(_i1), int(_i2), float(z_angles[array[_i1 + 1, _i2] - 1])) for _i1, _i2 in zip(*zero_indices)),
        key=lambda x: x[1],
    )
    return result
