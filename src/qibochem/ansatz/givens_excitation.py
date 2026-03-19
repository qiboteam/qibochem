"""
Circuit ansatz for representing a fermionic excitation as a Givens rotation by Arrazola et al.
"""

from qibo import Circuit, gates

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.util import generate_excitations, mp2_amplitude, sort_excitations

# Helper functions


def single_excitation_gate(sorted_orbitals, theta):
    """
    Decomposition of a Givens single excitation gate into single qubit rotations and CNOTs. In principle, should be
    identical to gates.GIVENS(qubit0, qubit1, theta)

    Args:
        sorted_orbitals (list): Sorted list of orbitals involved in the excitation
        theta (float): Rotation angle

    Returns:
        (list): List of gates representing the decomposition of the Givens' single excitation gate
    """
    result = []
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    result.append(gates.RY(sorted_orbitals[0], 0.5 * theta))
    result.append(gates.CNOT(sorted_orbitals[1], sorted_orbitals[0]))
    result.append(gates.RY(sorted_orbitals[0], -0.5 * theta))
    result.append(gates.CNOT(sorted_orbitals[1], sorted_orbitals[0]))
    result.append(gates.CNOT(sorted_orbitals[0], sorted_orbitals[1]))
    return result


def double_excitation_gate(sorted_orbitals, theta):
    """
    Decomposition of a Givens double excitation gate into single qubit rotations and CNOTs

    Args:
        sorted_orbitals (list): Sorted list of orbitals involved in the excitation
        theta (float): Rotation angle

    Returns:
        (list): List of gates representing the decomposition of the Givens' double excitation gate
    """
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
def givens_excitation_circuit(n_qubits, excitation, theta=None):
    """
    Circuit ansatz corresponding to the Givens rotation excitation from Arrazola et al.

    Args:
        n_qubits (int): Number of qubits in the circuit
        excitation (list): Iterable of orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta (float): Rotation angle. Default: 0.0

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit ansatz for a single Givens excitation

    References:
        1. J. M. Arrazola, O. D. Matteo, N. Quesada, S. Jahangiri, A. Delgado, and Nathan Killoran, *Universal quantum
        circuits for quantum chemistry*, Quantum, 2022, 6, 742.
        (`link <https://quantum-journal.org/papers/q-2022-06-20-742>`__)
    """
    sorted_orbitals = sorted(excitation)
    # Check size of orbitals input
    assert len(sorted_orbitals) % 2 == 0, f"{excitation} must have an even number of items"

    if theta is None:
        theta = 0.0

    circuit = Circuit(n_qubits)
    if len(excitation) == 2:
        circuit.add(single_excitation_gate(sorted_orbitals, theta))
    elif len(excitation) == 4:
        circuit.add(double_excitation_gate(sorted_orbitals, theta))
    else:
        raise NotImplementedError("Can only handle single and double excitations!")
    return circuit


def givens_excitation_ansatz(
    molecule,
    excitations=None,
    include_hf=True,
    use_mp2_guess=True,
):
    """
    Convenience function for buildng a circuit corresponding to the Givens excitation ansatz with multiple excitations
    for a given :class:`qibochem.driver.Molecule`. If no excitations are given, it defaults to including all possible
    spin-allowed excitations (up to doubles).

    Args:
        molecule (:class:`qibochem.driver.Molecule`): Molecule of interest
        excitations (list): List of excitations (e.g. ``[[0, 1, 2, 3], [0, 1, 4, 5]]``) used to build the
            Givens excitation circuit
        include_hf (bool): Whether or not to start the circuit with a Hartree-Fock circuit. Default: ``True``
        use_mp2_guess (bool): Whether to use MP2 amplitudes or a numpy zero array as the initial guess parameter.
            Default: ``True``, uses the MP2 amplitudes as the initial guess parameters

    Returns:
        :class:`qibo.models.Circuit`: Circuit corresponding to a Givens excitation circuit ansatz
    """
    # TODO: Consolidate/Meld this function with the ucc_ansatz function; both are largely identical

    # Get the number of electrons and spin-orbitals from the molecule argument
    n_elec = molecule.nelec if molecule.n_active_e is None else molecule.n_active_e
    n_orbs = molecule.nso if molecule.n_active_orbs is None else molecule.n_active_orbs

    # If no excitations given, defaults to all possible double and single excitations
    if excitations is None:
        excitations = []
        for order in range(2, 0, -1):  # Reversed to get double excitations first, then singles
            excitations += sort_excitations(generate_excitations(order, range(0, n_elec), range(n_elec, n_orbs)))
    else:
        # Some checks to ensure the given excitations are valid
        assert all(len(_ex) in (2, 4) for _ex in excitations), "Only single and double excitations allowed!"

    # Build the circuit
    if include_hf:
        circuit = hf_circuit(n_orbs, n_elec)  # Only works with (default) JW mapping
    else:
        circuit = Circuit(n_orbs)
    for excitation in excitations:
        theta = mp2_amplitude(excitation, molecule.eps, molecule.tei) if use_mp2_guess else None
        circuit += givens_excitation_circuit(n_orbs, excitation, theta)
    return circuit
