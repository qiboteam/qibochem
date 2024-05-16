"""
Symmetry-Preserving circuit ansatz from Gard et al. Reference: https://doi.org/10.1038/s41534-019-0240-1
"""

import numpy as np
from qibo import Circuit, gates

# Helper functions


def a_gate(qubit1, qubit2):
    """
    Returns the list of elementary gates corresponding to the 'A' gate as defined in the paper, acting on qubit1 and qubit2
    """
    result = []
    result.append(gates.CNOT(qubit2, qubit1))
    result += [gates.RY(qubit2, 0.0), gates.RZ(qubit2, 0.0)]
    result.append(gates.CNOT(qubit1, qubit2))
    result += [gates.RZ(qubit2, 0.0), gates.RY(qubit2, 0.0)]
    result.append(gates.CNOT(qubit2, qubit1))
    return result


# Main function
def symm_preserving_circuit(n_qubits, n_electrons):
    """
    Symmetry-preserving circuit ansatz

    Args:
        n_qubits: Number of qubits in the quantum circuit
        n_electrons: Number of electrons in the molecular system

    Returns:
        Qibo ``Circuit``: Circuit ansatz
    """
    circuit = Circuit(n_qubits)
    circuit.add(gates.X(2 * _i) for _i in range(n_electrons))

    a_gate_qubits = []  # Generate the list of qubits pairs for adding A gates

    a_gate_qubits = [(0, 1)]

    a_gates = [a_gate(qubit1, qubit2) for qubit1, qubit2 in a_gate_qubits]
    circuit.add(_gates for _a_gate in a_gates for _gates in _a_gate)  # Unpack the nested list

    return circuit


n_qubits = 4
n_electrons = 2
circuit = symm_preserving_circuit(n_qubits, n_electrons)
print(circuit.draw())
