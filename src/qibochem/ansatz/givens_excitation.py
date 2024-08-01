"""
Module documentation
"""

from qibo import Circuit, gates

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.util import generate_excitations, mp2_amplitude, sort_excitations

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


def givens_excitation_ansatz(
    molecule,
    excitation_level=None,
    excitations=None,
    include_hf=True,
    use_mp2_guess=True,
):
    """
    Convenience function for buildng a circuit corresponding to the Givens excitation ansatz with multiple excitations
    for a given ``Molecule``. If no excitations are given, it defaults to including all possible spin-allowed
    excitations, up to doubles.

    Args:
        molecule: The ``Molecule`` of interest.
        excitation_level: Include excitations up to how many electrons, i.e. ``"S"`` or ``"D"``.
            Ignored if ``excitations`` argument is given. Default: ``"D"``, i.e. double excitations
        excitations: List of excitations (e.g. ``[[0, 1, 2, 3], [0, 1, 4, 5]]``) used to build the
            UCC circuit. Overrides the ``excitation_level`` argument
        include_hf: Whether or not to start the circuit with a Hartree-Fock circuit. Default: ``True``
        use_mp2_guess: Whether to use MP2 amplitudes or a numpy zero array as the initial guess parameter. Default: ``True``;
            use the MP2 amplitudes as the default guess parameters

    Returns:
        Qibo ``Circuit``: Circuit corresponding to the 'Universal' ansatz
    """
    # TODO: Consolidate/Meld this function with the ucc_ansatz function; both are largely identical

    # Get the number of electrons and spin-orbitals from the molecule argument
    n_elec = molecule.nelec if molecule.n_active_e is None else molecule.n_active_e
    n_orbs = molecule.nso if molecule.n_active_orbs is None else molecule.n_active_orbs

    # Define the excitation level to be used if no excitations given
    if excitations is None:
        excitation_levels = ("S", "D", "T", "Q")
        if excitation_level is None:
            excitation_level = "D"
        else:
            # Check validity of input
            assert (
                len(excitation_level) == 1 and excitation_level.upper() in excitation_levels
            ), "Unknown input for excitation_level"
            # Note: Probably don't be too ambitious and try to do 'T'/'Q' at the moment...
            if excitation_level.upper() in ("T", "Q"):
                raise NotImplementedError("Cannot handle triple and quadruple excitations!")
        # Get the (largest) order of excitation to use
        excitation_order = excitation_levels.index(excitation_level.upper()) + 1

        # Generate and sort all the possible excitations
        excitations = []
        for order in range(excitation_order, 0, -1):  # Reversed to get higher excitations first
            excitations += sort_excitations(generate_excitations(order, range(0, n_elec), range(n_elec, n_orbs)))
    else:
        # Some checks to ensure the given excitations are valid
        assert all(len(_ex) % 2 == 0 for _ex in excitations), "Excitation with an odd number of elements found!"

    # Build the circuit
    if include_hf:
        circuit = hf_circuit(n_orbs, n_elec)  # Only works with (default) JW mapping
    else:
        circuit = Circuit(n_orbs)
    for excitation in excitations:
        theta = mp2_amplitude(excitation, molecule.eps, molecule.tei) if use_mp2_guess else None
        circuit += givens_excitation_circuit(n_orbs, excitation, theta)
    return circuit
