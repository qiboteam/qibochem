"""Helper functions for the `ansatz` module"""

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error


def _bk_matrix_power2(dims: int) -> np.ndarray:
    """Build the Bravyi-Kitaev matrix of dimension ``dims`` :math:`d = 2^{n}` recursively"""
    if dims < 1:
        raise_error(ValueError, "Dimension of Bravyi-Kitaev matrix must be at least 1")
    # Base case
    elif dims == 1:
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
    if dims < 1:
        raise_error(ValueError, "Dimension of Bravyi-Kitaev matrix must be at least 1")
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
