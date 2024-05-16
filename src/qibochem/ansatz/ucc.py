"""
Circuit representing the Unitary Coupled Cluster ansatz in quantum chemistry
"""

import numpy as np
import openfermion
from qibo import Circuit, gates

from qibochem.ansatz.hf_reference import hf_circuit


def expi_pauli(n_qubits, pauli_string, theta):
    """
    Build circuit representing exp(i*theta*pauli_string)

    Args:
        n_qubits: No. of qubits in the quantum circuit
        pauli_string: String in the format: ``"X0 Z1 Y3 X11"``
        theta: Real number

    Returns:
        circuit: Qibo Circuit object representing exp(i*theta*pauli_string)
    """
    # Split pauli_string into the old p_letters format
    pauli_ops = sorted(((int(_op[1:]), _op[0]) for _op in pauli_string.split()), key=lambda x: x[0])
    n_pauli_ops = len(pauli_ops)

    # Convert theta into a real number for applying with a RZ gate
    rz_parameter = -2.0 * theta

    # Generate the list of basis change gates using the pauli_ops list
    basis_changes = []
    for qubit, pauli_op in pauli_ops:
        if pauli_op == "Y":
            basis_changes.append(gates.S(qubit).dagger())
        if pauli_op not in ("I", "Z"):
            basis_changes.append(gates.H(qubit))

    # Build the circuit
    circuit = Circuit(n_qubits)
    # 1. Change to X/Y where necessary
    circuit.add(basis_changes)
    # 2. Add CNOTs to all pairs of qubits in pauli_ops, starting from the last letter
    circuit.add(gates.CNOT(pauli_ops[_i][0], pauli_ops[_i - 1][0]) for _i in range(n_pauli_ops - 1, 0, -1))
    # 3. Add RZ gate to last element of pauli_ops
    circuit.add(gates.RZ(pauli_ops[0][0], rz_parameter))
    # 4. Add CNOTs to all pairs of qubits in pauli_ops
    circuit.add(gates.CNOT(pauli_ops[_i + 1][0], pauli_ops[_i][0]) for _i in range(n_pauli_ops - 1))
    # 3. Change back to the Z basis
    circuit.add(_gate.dagger() for _gate in reversed(basis_changes))
    return circuit


def ucc_circuit(n_qubits, excitation, theta=0.0, trotter_steps=1, ferm_qubit_map=None):
    r"""
    Circuit corresponding to the unitary coupled-cluster ansatz for a single excitation

    Args:
        n_qubits: Number of qubits in the quantum circuit
        excitation: Iterable of orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta: UCC parameter. Defaults to 0.0
        trotter_steps: Number of Trotter steps; i.e. number of times the UCC ansatz is applied with ``theta`` = ``theta / trotter_steps``. Default: 1
        ferm_qubit_map: Fermion-to-qubit transformation. Default is Jordan-Wigner (``jw``).

    Returns:
        Qibo ``Circuit``: Circuit corresponding to a single UCC excitation
    """
    # Check size of orbitals input
    n_orbitals = len(excitation)
    assert n_orbitals % 2 == 0, f"{excitation} must have an even number of items"
    # Reverse sort orbitals to get largest-->smallest
    sorted_orbitals = sorted(excitation, reverse=True)

    # Define default mapping
    if ferm_qubit_map is None:
        ferm_qubit_map = "jw"

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
    else:
        raise KeyError("Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Apply the qubit_ucc_operator 'trotter_steps' times:
    assert trotter_steps > 0, f"{trotter_steps} must be > 0!"
    circuit = Circuit(n_qubits)
    for _i in range(trotter_steps):
        # Use the get_operators() generator to get the list of excitation operators
        for raw_pauli_string in qubit_ucc_operator.get_operators():
            # Convert each operator into a string and get the associated coefficient
            ((pauli_ops, coeff),) = raw_pauli_string.terms.items()  # Unpack the single-item dictionary
            pauli_string = " ".join(f"{pauli_op[1]}{pauli_op[0]}" for pauli_op in pauli_ops)
            # Build the circuit and add it on
            _circuit = expi_pauli(
                n_qubits, pauli_string, -1.0j * coeff * theta / trotter_steps
            )  # Divide imag. coeff by 1.0j
            circuit += _circuit
    return circuit


# Utility functions


def mp2_amplitude(excitation, orbital_energies, tei):
    r"""
    Calculate the MP2 guess amplitude for a single UCC circuit: 0.0 for a single excitation.
        for a double excitation (In SO basis): :math:`t_{ij}^{ab} = (g_{ijab} - g_{ijba}) / (e_i + e_j - e_a - e_b)`

    Args:
        excitation: Iterable of spin-orbitals representing a excitation. Must have either 2 or 4 elements exactly,
            representing a single or double excitation respectively.
        orbital_energies: eigenvalues of the Fock operator, i.e. orbital energies
        tei: Two-electron integrals in MO basis and second quantization notation

    Returns:
        MP2 guess amplitude (float)
    """
    # Checks validity of excitation argument
    assert len(excitation) % 2 == 0 and len(excitation) // 2 <= 2, f"{excitation} must have either 2 or 4 elements"
    # If single excitation, can just return 0.0 directly
    if len(excitation) == 2:
        return 0.0

    # Convert orbital indices to be in MO basis
    mo_orbitals = [orbital // 2 for orbital in excitation]
    # Numerator: g_ijab - g_ijba
    g_ijab = (
        tei[tuple(mo_orbitals)]  # Can index directly using the MO TEIs
        if (excitation[0] + excitation[3]) % 2 == 0 and (excitation[1] + excitation[2]) % 2 == 0
        else 0.0
    )
    g_ijba = (
        tei[tuple(mo_orbitals[:2] + mo_orbitals[2:][::-1])]  # Reverse last two terms
        if (excitation[0] + excitation[2]) % 2 == 0 and (excitation[1] + excitation[3]) % 2 == 0
        else 0.0
    )
    numerator = g_ijab - g_ijba
    # Denominator is directly from the orbital energies
    denominator = sum(orbital_energies[mo_orbitals[:2]]) - sum(orbital_energies[mo_orbitals[2:]])
    return numerator / denominator


def generate_excitations(order, excite_from, excite_to, conserve_spin=True):
    """
    Generate all possible excitations between a list of occupied and virtual orbitals

    Args:
        order: Order of excitations, i.e. 1 == single, 2 == double
        excite_from: Iterable of integers
        excite_to: Iterable of integers
        conserve_spin: ensure that the total electronic spin is conserved

    Return:
        List of lists, e.g. [[0, 1]]
    """
    # If order of excitation > either number of electrons/orbitals, return list of empty list
    if order > min(len(excite_from), len(excite_to)):
        return [[]]

    # Generate all possible excitations first
    from itertools import combinations  # pylint: disable=C0415

    all_excitations = [
        [*_from, *_to] for _from in combinations(excite_from, order) for _to in combinations(excite_to, order)
    ]
    # Filter out the excitations if conserve_spin set
    if conserve_spin:
        # Not sure if this filtering is exhaustive; might not remove some redundant excitations?
        all_excitations = [
            _ex
            for _ex in all_excitations
            if sum(_ex) % 2 == 0 and (sum(_i % 2 for _i in _ex[:order]) == sum(_i % 2 for _i in _ex[order:]))
        ]
    return all_excitations


def sort_excitations(excitations):
    """
    Sorts a list of excitations according to some common-sense and empirical rules (see below).
        The order of the excitations must be the same throughout.

    Sorting order:
    1. (For double excitations only) All paired excitations between the same MOs first
    2. Pair up excitations between the same MOs, e.g. (0, 2) and (1, 3)
    3. Then count upwards from smallest MO

    Args:
        excitations: List of iterables, each representing an excitation; e.g. [[1, 5], [0, 4]]

    Returns:
        List of excitations after sorting
    """
    # Check that all excitations are of the same order and <= 2
    order = len(excitations[0]) // 2
    if order > 2:
        raise NotImplementedError("Can only handle single and double excitations!")
    assert all(len(_ex) // 2 == order for _ex in excitations), "Cannot handle excitations of different orders!"

    # Define variables for the while loop
    copy_excitations = [list(_ex) for _ex in excitations]
    result = []
    prev = []

    # No idea how I came up with this, but it seems to work for double excitations
    def sorting_fn(x):
        # Default sorting is OK for single excitations
        return sum((order + 1 - _i) * abs(x[2 * _i + 1] // 2 - x[2 * _i] // 2) for _i in range(0, order))

    # Make a copy of the list of excitations, and use it populate a new list iteratively
    while copy_excitations:
        if not prev:
            # Take out all pair excitations first
            pair_excitations = [
                _ex
                for _ex in copy_excitations
                # Indices of the electrons/holes must be consecutive numbers
                if sum(abs(_ex[2 * _i + 1] // 2 - _ex[2 * _i] // 2) for _i in range(0, order)) == 0
            ]
            while pair_excitations:
                pair_excitations = sorted(pair_excitations)
                ex_to_remove = pair_excitations.pop(0)
                if ex_to_remove in copy_excitations:
                    # 'Move' the first pair excitation from copy_excitations to result
                    index = copy_excitations.index(ex_to_remove)
                    result.append(copy_excitations.pop(index))

            # No more pair excitations, only remaining excitations should have >=3 MOs involved
            # Sort the remaining excitations
            copy_excitations = sorted(copy_excitations, key=sorting_fn if order != 1 else None)
        else:
            # Check to see for excitations involving the same MOs as prev
            _from = prev[:order]
            _to = prev[order:]
            # Get all possible excitations involving the same MOs
            new_from = [_i + 1 if _i % 2 == 0 else _i - 1 for _i in _from]
            new_to = [_i + 1 if _i % 2 == 0 else _i - 1 for _i in _to]
            same_mo_ex = [sorted(list(_f) + list(_t)) for _f in (_from, new_from) for _t in (_to, new_to)]
            # Remove the excitations with the same MOs from copy_excitations
            while same_mo_ex:
                same_mo_ex = sorted(same_mo_ex)
                ex_to_remove = same_mo_ex.pop(0)
                if ex_to_remove in copy_excitations:
                    # 'Move' the first entry of same_mo_index from copy_excitations to result
                    index = copy_excitations.index(ex_to_remove)
                    result.append(copy_excitations.pop(index))
            prev = None
            continue
        if copy_excitations:
            # Remove the first entry from the sorted list of remaining excitations and add it to result
            prev = copy_excitations.pop(0)
            result.append(prev)
    return result


def ucc_ansatz(
    molecule,
    excitation_level=None,
    excitations=None,
    thetas=None,
    trotter_steps=1,
    ferm_qubit_map=None,
    include_hf=True,
    use_mp2_guess=True,
):
    """
    Convenience function for buildng a circuit corresponding to the UCC ansatz with multiple excitations for a given ``Molecule``.
    If no excitations are given, it defaults to returning the full UCCSD circuit ansatz.

    Args:
        molecule: The ``Molecule`` of interest.
        excitation_level: Include excitations up to how many electrons, i.e. ``"S"`` or ``"D"``.
            Ignored if ``excitations`` argument is given. Default: ``"D"``, i.e. double excitations
        excitations: List of excitations (e.g. ``[[0, 1, 2, 3], [0, 1, 4, 5]]``) used to build the
            UCC circuit. Overrides the ``excitation_level`` argument
        thetas: Parameters for the excitations. Default value depends on the ``use_mp2_guess`` argument.
        trotter_steps: number of Trotter steps; i.e. number of times the UCC ansatz is applied with
            ``theta`` = ``theta / trotter_steps``. Default: 1
        ferm_qubit_map: fermion-to-qubit transformation. Default: Jordan-Wigner (``"jw"``)
        include_hf: Whether or not to start the circuit with a Hartree-Fock circuit. Default: ``True``
        use_mp2_guess: Whether to use MP2 amplitudes or a numpy zero array as the initial guess parameter. Default: ``True``;
            use the MP2 amplitudes as the default guess parameters

    Returns:
        Qibo ``Circuit``: Circuit corresponding to an UCC ansatz
    """
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

    # Check if thetas argument given, define to be all zeros if not
    # Number of circuit parameters: S->2, D->8, (T/Q->32/128; Not sure?)
    n_parameters = 2 * len([_ex for _ex in excitations if len(_ex) == 2])  # Singles
    n_parameters += 8 * len([_ex for _ex in excitations if len(_ex) == 4])  # Doubles
    if thetas is None:
        if use_mp2_guess:
            thetas = np.array([mp2_amplitude(excitation, molecule.eps, molecule.tei) for excitation in excitations])
        else:
            thetas = np.zeros(n_parameters)
    else:
        # Check that number of circuit variables (i.e. thetas) matches the number of circuit parameters
        assert len(thetas) == n_parameters, "Number of input parameters doesn't match the number of circuit parameters!"

    # Build the circuit
    if include_hf:
        circuit = hf_circuit(n_orbs, n_elec, ferm_qubit_map=ferm_qubit_map)
    else:
        circuit = Circuit(n_orbs)
    for excitation, theta in zip(excitations, thetas):
        circuit += ucc_circuit(n_orbs, excitation, theta, trotter_steps=trotter_steps, ferm_qubit_map=ferm_qubit_map)
    return circuit
