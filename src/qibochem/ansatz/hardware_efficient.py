"""
Hardware efficient ansatz, e.g. Kandala et al. (https://doi.org/10.1038/nature23879)
"""

from qibo import Circuit, gates
from qibo.models.encodings import entangling_layer


def he_circuit(
    n_qubits,
    n_layers,
    parameter_gates=None,
    entangling_gate="CNOT",
    architecture="diagonal",
    closed_boundary=True,
    **kwargs
):
    """
    Builds a general hardware-efficient ansatz, in which the rotation and entangling gates used can be chosen by the
    user. For more details on the arguments related to the entangling layer, see the documentation for
    :class:`qibo.models.encodings.entangling_layer`.

    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        n_layers (int): Number of layers of rotation and entangling gates
        parameter_gates (list): Iterable of single-qubit rotation gates to be used in the ansatz. The gates should be
            given either as strings representing valid one-qubit gates, or as :class:`qibo.gates.Gate` directly.
            Default: ``["RY", "RZ"]``
        entangling_gate (str or :qibo.gates.Gate:, optional): String representing the two-qubit entangling gate to be
            used in the ansatz; should be a valid two-qubit gate. Default: ``"CNOT"``
        architecture (str, optional): Architecture of the entangling layer, with the possible options: ``"diagonal"``,
            ``"even_layer"``, ``"next_nearest"``, ``"odd_layer"``, ``"pyramid"``, ``"shifted"``, ``"v"``, and ``"x"``
            (defined only for an even number of qubits. Default: ``"diagonal"``.
        closed_boundary (bool, optional): If ``True`` and ``architecture not in ["pyramid", "v", "x"]``, adds a
            closed-boundary condition to the entangling layer. Default: ``True``
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object. Details are given in the
            documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit corresponding to the hardware-efficient ansatz
    """
    # Default variables
    if parameter_gates is None:
        parameter_gates = ["RY", "RZ"]
    parameter_gates = [getattr(gates, _gate) if isinstance(_gate, str) else _gate for _gate in parameter_gates]

    circuit = Circuit(n_qubits, **kwargs)
    for _ in range(n_layers):
        # Rotation gates
        circuit.add(rgate(qubit, theta=0.0) for qubit in range(n_qubits) for rgate in parameter_gates)
        # Entangling gates
        circuit += entangling_layer(n_qubits, architecture, entangling_gate, closed_boundary, **kwargs)
    return circuit
