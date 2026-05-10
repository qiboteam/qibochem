"""
Circuit ansatzes for chemistry
"""

from collections.abc import Sequence

import numpy as np
import openfermion
from qibo import Circuit, gates
from qibo.config import raise_error
from qibo.gates import Gate
from qibo.models.encodings import comp_basis_encoder, entangling_layer

from qibochem.ansatz._ansatz import _bk_matrix, _expi_pauli


def he_circuit(
    nqubits: int,
    nlayers: int,
    rotation_gates: Sequence[str | Gate] | None = None,
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
        rotation_gates (Sequence[str | Gate] | None, optional): Single-qubit rotation gates used in the ansatz. These
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
            documentation of :class:`qibo.models.circuit.Circuit`

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit corresponding to the hardware-efficient ansatz
    """
    # Default variables
    if rotation_gates is None:
        rotation_gates = ["RY", "RZ"]
    # if not all(isinstance(_gate, (str, Gate)) for _gate in rotation_gates):
    #     raise_error(TypeError, "Invalid one-qubit rotation gate input")
    rotation_gates = [getattr(gates, _gate) if isinstance(_gate, str) else _gate for _gate in rotation_gates]

    circuit = Circuit(nqubits, **kwargs)
    for _ in range(nlayers):
        # Rotation gates
        circuit.add(
            rgate(qubit, theta=0.0)  # pylint: disable=not-callable
            for qubit in range(nqubits)
            for rgate in rotation_gates
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


def ucc_circuit(
    nqubits: int,
    excitation: Sequence[int],
    theta: float = 0.0,
    trotter_steps: int = 1,
    ferm_qubit_map: str | None = None,
    **kwargs: dict,
) -> Circuit:
    r"""
    Circuit corresponding to the unitary coupled-cluster ansatz for a single excitation

    Args:
        nqubits (int): Number of qubits in the quantum circuit
        excitation (Sequence[int]): Orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta (float, optional): UCC parameter. Defaults to 0.0
        trotter_steps (int, optional): Number of Trotter steps; i.e. number of times the UCC ansatz is applied
            with :math:`\theta = \theta` / ``trotter_steps``. Default: 1
        ferm_qubit_map (str, optional): Fermion-to-qubit transformation. Default: Jordan-Wigner (``"jw"``)
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object. Details are given in the
            documentation of :class:`qibo.models.circuit.Circuit`

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit corresponding to a single UCC excitation
    """
    # Check size of orbitals input
    n_orbitals = len(excitation)
    if not n_orbitals:
        raise_error(ValueError, f"No excitations given")
    if n_orbitals % 2 != 0:
        raise_error(ValueError, f"{excitation} must have an even number of items")
    # Reverse sort orbitals to get largest-->smallest
    sorted_orbitals = sorted(excitation, reverse=True)

    # Define default mapping and check input is valid
    if ferm_qubit_map is None:
        ferm_qubit_map = "jw"
    if ferm_qubit_map not in ("jw", "bk"):
        raise_error(NotImplementedError, "Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Define the UCC excitation operator corresponding to the given list of orbitals
    fermion_op_str_template = f"{(n_orbitals//2)*'{}^ '}{(n_orbitals//2)*'{} '}"
    fermion_operator_str = fermion_op_str_template.format(*sorted_orbitals)
    # Build the FermionOperator and make it unitary
    fermion_operator = openfermion.FermionOperator(fermion_operator_str)
    ucc_operator = fermion_operator - openfermion.hermitian_conjugated(fermion_operator)

    # Map the FermionOperator to a QubitOperator
    if ferm_qubit_map == "jw":
        qubit_ucc_operator = openfermion.jordan_wigner(ucc_operator)
    elif ferm_qubit_map == "bk":
        qubit_ucc_operator = openfermion.bravyi_kitaev(ucc_operator)

    # Apply the qubit_ucc_operator 'trotter_steps' times:
    if trotter_steps < 1:
        raise_error(ValueError, f"{trotter_steps} must be > 0!")
    circuit = Circuit(nqubits, **kwargs)
    for _i in range(trotter_steps):
        for pauli_ops, coeff in qubit_ucc_operator.terms.items():
            # Convert each operator into a string and get the associated coefficient
            pauli_string = " ".join(f"{pauli_op[1]}{pauli_op[0]}" for pauli_op in pauli_ops)
            # Build the circuit and add it on
            circuit += _expi_pauli(
                nqubits, pauli_string, -1.0j * coeff * theta / trotter_steps
            )  # Divide imag. coeff by 1.0j
    return circuit


def qeb_circuit(nqubits: int, excitation: Sequence[int], theta: float = 0.0, **kwargs: dict) -> Circuit:
    r"""
    Qubit-excitation-based (QEB) circuit corresponding to the unitary coupled-cluster ansatz for a single excitation.
    This circuit ansatz is only valid for the Jordan-Wigner fermion to qubit mapping.

    Args:
        nqubits (int): Number of qubits in the quantum circuit
        excitation (Sequence[int]): Orbitals involved in the excitation; must have an even number of elements.
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta (float, optional): UCC parameter. Defaults to 0.0
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object. Details are given in the
            documentation of :class:`qibo.models.circuit.Circuit`

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit corresponding to a single UCC excitation

    References:
        1. I. Magoulas and F. A. Evangelista, *CNOT-Efficient Circuits for Arbitrary Rank Many-Body Fermionic and Qubit
        Excitations*, Journal of Chemical Theory and Computation, 2023, 19(3), 822-836.
        (links: `here <https://pubs.acs.org/doi/10.1021/acs.jctc.2c01016>`__ or
        on `arXiv <https://arxiv.org/abs/2210.05771>`__)
    """
    n_orbitals = len(excitation)
    if not n_orbitals:
        raise_error(ValueError, f"No excitations given")
    if n_orbitals % 2 != 0:
        raise_error(ValueError, f"{excitation} must have an even number of items")

    n_tuples = len(excitation) // 2
    i_array = excitation[:n_tuples]
    a_array = excitation[n_tuples:]
    fwd_gates = [gates.CNOT(i_array[-1], _i) for _i in i_array[-2::-1]]
    fwd_gates += [gates.CNOT(a_array[-1], _a) for _a in a_array[-2::-1]]
    fwd_gates.append(gates.CNOT(a_array[-1], i_array[-1]))
    fwd_gates += [gates.X(_ia) for _ia in excitation if _ia not in (i_array[-1], a_array[-1])]
    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gate for gate in fwd_gates)
    # MCRY
    # multi-controlled RY gate,
    # negative controls i, a
    # positive control on i_n
    mcry_controls = excitation[:-1]
    ry_angle = 2.0 * theta
    circuit.add(gates.RY(a_array[-1], ry_angle).controlled_by(*mcry_controls))
    circuit.add(gate for gate in fwd_gates[::-1])
    return circuit


def givens_excitation_circuit(nqubits: int, excitation: Sequence[int], theta: float = 0.0) -> Circuit:
    """
    Circuit ansatz corresponding to the Givens rotation from Arrazola et al.

    Args:
        nqubits (int): Number of qubits in the circuit
        excitation (Sequence[int]): Iterable of orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta (float, optional): Rotation angle. Default: 0.0

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit ansatz for a single Givens rotation

    References:
        1. J. M. Arrazola, O. D. Matteo, N. Quesada, S. Jahangiri, A. Delgado, and Nathan Killoran, *Universal quantum
        circuits for quantum chemistry*, Quantum, 2022, 6, 742.
        (`link <https://quantum-journal.org/papers/q-2022-06-20-742>`__)
    """
    sorted_orbitals = sorted(excitation)
    n_orbitals = len(excitation)
    # Check excitation input
    if not n_orbitals:
        raise_error(ValueError, f"No excitations given")
    if n_orbitals % 2 != 0:
        raise_error(ValueError, f"{excitation} must have an even number of items")
    qubits_in, qubits_out = sorted_orbitals[: (n_orbitals // 2)], sorted_orbitals[(n_orbitals // 2) :]

    circuit = Circuit(nqubits)
    if n_orbitals == 2:
        circuit.add(gates.GIVENS(qubits_in[0], qubits_out[0], theta))
    else:
        circuit.add(gates.GeneralizedRBS(qubits_in, qubits_out, -theta))  # phi parameter not used here
    return circuit
