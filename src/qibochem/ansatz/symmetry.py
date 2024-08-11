"""
Symmetry-Preserving circuit ansatz from Gard et al. Reference: https://doi.org/10.1038/s41534-019-0240-1
"""

from math import factorial

import numpy as np
from qibo import Circuit, gates

# Helper functions


def a_gate(qubit1, qubit2, theta=None, phi=None):
    """
    Decomposition of the 'A' gate as defined in the paper, acting on qubit1 and qubit2. 'A' corresponds to the following
    unitary matrix:

    A(\\theta, \\phi) =
    \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\cos \\theta & e^{i \\phi} \\sin \\theta & 0 \\\\
        0 & e^{-i \\phi} \\sin \\theta & -\\cos \\theta & 0 \\\\
        0 & 0 & 0 & 1
    \\end{pmatrix}

    Args:
        qubit1 (int): Index of the first qubit
        qubit2 (int): Index of the second qubit
        theta (float): First rotation angle. Default: 0.0
        phi (float): Second rotation angle. Default: 0.0

    Returns:
        (list): List of gates representing the decomposition of the 'A' gate
    """
    if theta is None:
        theta = 0.0
    if phi is None:
        phi = 0.0

    # R(theta, phi) = R_z (phi + pi) R_y (theta + 0.5*pi)
    r_gate = [gates.RY(qubit2, theta + 0.5 * np.pi), gates.RZ(qubit2, phi + np.pi)]
    r_gate_dagger = [_gate.dagger() for _gate in r_gate][::-1]
    return (
        [gates.CNOT(qubit2, qubit1)]
        + r_gate_dagger
        + [gates.CNOT(qubit1, qubit2)]
        + r_gate
        + [gates.CNOT(qubit2, qubit1)]
    )


def x_gate_indices(n_qubits, n_electrons):
    """Obtain the qubit indices for X gates to be added to the circuit"""
    indices = [2 * _i for _i in range(0, min(n_electrons, n_qubits // 2))]
    if n_electrons > n_qubits // 2:
        indices += [2 * _i + 1 for _i in range(n_electrons - (n_qubits // 2))]
    return sorted(indices)


def a_gate_indices(n_qubits, n_electrons, x_gates):
    """
    Obtain the qubit indices for a single layer of the primitive pattern of 'A' gates in the circuit ansatz
    """
    assert len(x_gates) == n_electrons, f"n_electrons ({n_electrons}) != Number of X gates given! ({x_gates})"
    # 2. Apply 'first layer' of gates on all adjacent pairs of qubits on which either X*I or I*X has been applied.
    first_layer = [(_i, _i + 1) for _i in x_gates if _i + 1 < n_qubits and _i + 1 not in x_gates]
    first_layer += [(_i - 1, _i) for _i in x_gates if _i - 1 >= 0 and _i - 1 not in x_gates]
    # 3a. Apply 'second layer' of gates on adjacent pairs of qubits. Each pair includes 1 qubit acted on in the previous
    # step and a qubit free of gates. Continue placing gates on adjacent qubits until all neighboring qubits are connected
    second_layer = [(_i, _i + 1) for _i in range(max(pair[1] for pair in first_layer), n_qubits - 1)]
    second_layer += [(_i - 1, _i) for _i in range(min(pair[0] for pair in first_layer), 0, -1)]
    # 3b. The first and second layers define a primitive pattern:
    primitive_pattern = first_layer + second_layer
    # Need to add any missing connections between neighbouring qubits
    primitive_pattern += [pair for _i in range(n_qubits - 1) if (pair := (_i, _i + 1)) not in primitive_pattern]
    # 4. Repeat the primitive pattern until (n_qubits choose n_electrons) A gates are placed
    n_gates_per_layer = len(primitive_pattern)
    n_a_gates = factorial(n_qubits) // (factorial(n_qubits - n_electrons) * factorial(n_electrons))
    assert (
        n_a_gates % n_gates_per_layer == 0
    ), f"n_a_gates ({n_a_gates}) is not a multiple of n_gates_per_layer ({n_gates_per_layer})!"
    return (n_a_gates // n_gates_per_layer) * primitive_pattern


# Main function
def symm_preserving_circuit(n_qubits, n_electrons):
    """
    Symmetry-preserving circuit ansatz from Gard et al. (https://doi.org/10.1038/s41534-019-0240-1)

    Args:
        n_qubits: Number of qubits in the quantum circuit
        n_electrons: Number of electrons in the molecular system

    Returns:
        Qibo ``Circuit``: Circuit corresponding to the symmetry-preserving ansatz
    """
    circuit = Circuit(n_qubits)
    x_gates = x_gate_indices(n_qubits, n_electrons)
    circuit.add(gates.X(_i) for _i in x_gates)
    # Generate the qubit pair indices for adding A gates
    a_gate_qubits = a_gate_indices(n_qubits, n_electrons, x_gates)
    a_gates = [a_gate(qubit1, qubit2) for qubit1, qubit2 in a_gate_qubits]
    # Each a_gate is a list of elementary gates, so a_gates is a nested list; need to unpack it
    circuit.add(_gates for _a_gate in a_gates for _gates in _a_gate)
    return circuit
