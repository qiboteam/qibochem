from qibo import gates


def hea(n_layers, n_qubits, parameter_gates=["RY", "RZ"], coupling_gates="CZ"):
    """
    Builds the generalized hardware-efficient ansatz, in which the rotation and entangling gates used can be
    chosen by the user

    Args:
        n_layers: Number of layers of rotation and entangling gates
        n_qubits: Number of qubits in the quantum circuit
        parameter_gates: List of single-qubit rotation gates to be used in the ansatz. The gates should be given as
            strings representing valid ``Qibo`` one-qubit gates. Default: ``["RY", "RZ"]``
        coupling_gates: String representing the two-qubit entangling gate to be used in the ansatz; should be a
            valid two-qubit ``Qibo`` gate. Default: ``"CZ"``

    Returns:
        List of gates corresponding to the hardware-efficient ansatz
    """
    gate_list = []

    for ilayer in range(n_layers):
        for rgate in parameter_gates:
            for iqubit in range(n_qubits):
                gate_list.append(getattr(gates, rgate)(iqubit, theta=0))

        # entanglement
        for iqubit in range(n_qubits - 1):
            gate_list.append(getattr(gates, coupling_gates)(iqubit, iqubit + 1))
        gate_list.append(getattr(gates, coupling_gates)(n_qubits - 1, 0))

    return gate_list
