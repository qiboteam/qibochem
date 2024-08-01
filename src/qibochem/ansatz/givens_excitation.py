"""
Module documentation
"""

from qibo import Circuit, gates

# Helper functions


def double_excitation_gate(sorted_orbitals, theta=0.0):
    """
    Decomposition of a Givens double excitation gate into single qubit rotations and CNOTs

    Args:
        sorted_orbitals (list): Sorted list of orbitals involved in the excitation
        theta (float): Rotation angle. Default: 0.0

    Returns:
        (list): List of gates representing the decomposition of the Givens' double excitation gate
    """
    if theta is None:
        theta = 0.0

    result = []
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[2]))
    result.append(gates.H(sorted_orbitals[0]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[3]))
    result.append(gates.RY(sorted_orbitals[0], -0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], 0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[3]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[3], sorted_orbitals[1]))
    result.append(gates.RY(sorted_orbitals[0], -0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], 0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[1]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[0]))
    result.append(gates.RY(sorted_orbitals[0], 0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], -0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[3], sorted_orbitals[1]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[3]))
    result.append(gates.RY(sorted_orbitals[0], 0.125 * theta))
    result.append(gates.RY(sorted_orbitals[1], -0.125 * theta))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[0]))
    result.append(gates.H(sorted_orbitals[0]))
    result.append(gates.H(sorted_orbitals[3]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[2]))
    result.append(gates.CNOT(sorted_orbitals[2], sorted_orbitals[3]))
    return result


# Main function
def givens_excitation_circuit(n_qubits, excitation, theta=0.0):
    """
    Circuit ansatz for one Givens rotation excitation from Arrazola et al. Reference:
    https://doi.org/10.22331/q-2022-06-20-742

    Args:
        n_qubits: Number of qubits in the quantum circuit
        excitation: Iterable of orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``

    Returns:
        Qibo ``Circuit``: Circuit ansatz
    """
    sorted_orbitals = sorted(excitation)
    # Check size of orbitals input
    assert len(sorted_orbitals) % 2 == 0, f"{excitation} must have an even number of items"

    circuit = Circuit(n_qubits)
    if len(excitation) == 2:
        circuit.add(gates.GIVENS(excitation[0], excitation[1], theta))
    elif len(excitation) == 4:
        circuit.add(double_excitation_gate(sorted_orbitals, theta))
    else:
        raise NotImplementedError("Can only handle single and double excitations!")
    return circuit


def givens_excitation_ansatz(n_qubits, excitation, n_electrons):
    """TODO: Same implementation as UCC?"""
    pass
