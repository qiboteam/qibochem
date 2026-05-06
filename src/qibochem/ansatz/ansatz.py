"""
Circuit ansatzes for chemistry
"""

from collections.abc import Sequence

from qibo import Circuit, gates
from qibo.config import raise_error
from qibo.gates import Gate
from qibo.models.encodings import comp_basis_encoder, entangling_layer

from qibochem.ansatz._ansatz import _bk_matrix


def he_circuit(
    nqubits: int,
    nlayers: int,
    parameter_gates: Iterable[str | Gate] | None = None,
    entangling_gate: str | Gate = "CNOT",
    architecture: str = "diagonal",
    closed_boundary: bool = True,
    **kwargs,
) -> Circuit:
    """
    Builds a general hardware-efficient ansatz, in which the rotation and entangling gates used can be chosen by the
    user. For more details on the arguments related to the entangling layer, see the documentation for
    :class:`qibo.models.encodings.entangling_layer`.

    Args:
        nqubits (int): Number of qubits in the quantum circuit.
        nlayers (int): Number of layers of rotation and entangling gates.
        parameter_gates (Iterable[str | Gate] | None, optional): Single-qubit rotation gates used in the ansatz. These
            can be given as strings representing valid one-qubit gates, or as :class:`qibo.gates.Gate` directly.
            Default: ``["RY", "RZ"]``
        entangling_gate (str | Gate, optional): Two-qubit entangling gate used in the ansatz. This can be given as
            strings representing valid two-qubit gates, or as a :class:`qibo.gates.Gate` directly. Default: ``"CNOT"``
        architecture (str, optional): Architecture of the entangling layer, with the possible options: ``"diagonal"``,
            ``"even_layer"``, ``"next_nearest"``, ``"odd_layer"``, ``"pyramid"``, ``"shifted"``, ``"v"``, and ``"x"``
            (defined only for an even number of qubits. Default: ``"diagonal"``
        closed_boundary (bool, optional): If ``True`` (default) and ``architecture not in ["pyramid", "v", "x"]``, adds
            a closed-boundary condition to the entangling layer
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object. Details are given in the
            documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit corresponding to the hardware-efficient ansatz
    """
    # Default variables
    if parameter_gates is None:
        parameter_gates = ["RY", "RZ"]
    parameter_gates = [getattr(gates, _gate) if isinstance(_gate, str) else _gate for _gate in parameter_gates]

    circuit = Circuit(nqubits, **kwargs)
    for _ in range(nlayers):
        # Rotation gates
        circuit.add(
            rgate(qubit, theta=0.0)  # pylint: disable=not-callable
            for qubit in range(nqubits)
            for rgate in parameter_gates
        )
        # Entangling gates
        circuit += entangling_layer(nqubits, architecture, entangling_gate, closed_boundary, **kwargs)
    return circuit


def hf_circuit(nqubits: int, nelectrons: int, ferm_qubit_map: str | None = None, **kwargs) -> Circuit:
    """Circuit to prepare a Hartree-Fock state

    Args:
        nqubits (int): Number of qubits in the quantum circuit
        nelectrons (int): Number of electrons in the molecular system
        ferm_qubit_map (str | None, optional): Fermion to qubit map. Must be either Jordan-Wigner (``"jw"``) or
            Brayvi-Kitaev (``"bk"``). Default value is ``"jw"``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object. Details are given in the
            documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit initialized in a HF reference state
    """
    # Which fermion-to-qubit map to use
    if ferm_qubit_map is None:
        ferm_qubit_map = "jw"
    if ferm_qubit_map not in ("jw", "bk"):
        raise_error(KeyError, "Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Occupation number of SOs
    mapped_occ_n = None
    occ_n = np.concatenate((np.ones(nelectrons, dtype=np.int8), np.zeros(nqubits - nelectrons, dtype=np.int8)))
    if ferm_qubit_map == "jw":
        mapped_occ_n = occ_n
    elif ferm_qubit_map == "bk":
        mapped_occ_n = (_bk_matrix(nqubits) @ occ_n) % 2
    # Convert the array to a list, then build/return the final circuit
    return comp_basis_encoder(mapped_occ_n.tolist(), nqubits=nqubits, **kwargs)
