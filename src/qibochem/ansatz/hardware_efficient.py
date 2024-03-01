"""
Hardware efficient ansatz, e.g. Kandala et al. (https://doi.org/10.1038/nature23879)
"""

from qibo import Circuit, gates


def he_circuit(n_qubits, n_layers, parameter_gates=None, coupling_gates="CZ"):
    """
    Builds the generalized hardware-efficient ansatz, in which the rotation and entangling gates used can be
    chosen by the user

    Args:
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of layers of rotation and entangling gates
        parameter_gates: Iterable of single-qubit rotation gates to be used in the ansatz. The gates should be given as
            strings representing valid ``Qibo`` one-qubit gates. Default: ``["RY", "RZ"]``
        coupling_gates: String representing the two-qubit entangling gate to be used in the ansatz; should be a
            valid two-qubit ``Qibo`` gate. Default: ``"CZ"``

    Returns:
        Qibo ``Circuit``: Circuit corresponding to the hardware-efficient ansatz
    """
    # Default variables
    if parameter_gates is None:
        parameter_gates = ["RY", "RZ"]

    circuit = Circuit(n_qubits)

    for _ in range(n_layers):
        # Rotation gates
        circuit.add(getattr(gates, rgate)(qubit, theta=0.0) for qubit in range(n_qubits) for rgate in parameter_gates)

        # Entanglement gates
        circuit.add(getattr(gates, coupling_gates)(qubit, qubit + 1) for qubit in range(n_qubits - 1))
        circuit.add(getattr(gates, coupling_gates)(n_qubits - 1, 0))

    return circuit
