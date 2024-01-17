"""
Circuit representing the Unitary Coupled Cluster ansatz in quantum chemistry
"""

import numpy as np
import openfermion
from qibo import gates, models


def expi_pauli(n_qubits, theta, pauli_string):
    """
    Build circuit representing exp(i*theta*pauli_string)

    Args:
        n_qubits: No. of qubits in the quantum circuit
        theta: parameter
        pauli_string: OpenFermion QubitOperator object, e.g. X_0 Y_1 Z_2 X_4

    Returns:
        circuit: Qibo Circuit object representing exp(i*theta*pauli_string)
        coeff: Coefficient of theta. May be useful for VQE
    """
    # Unpack the dictionary from pauli_string.terms.items() into
    # a (tuple of Pauli letters) and the coefficient of the pauli_string
    [(p_letters, _coeff)] = pauli_string.terms.items()

    # _coeff is an imaginary number, i.e. exp(i\theta something)
    # Convert to the real coefficient for multiplying theta
    coeff = -2.0 * np.real(_coeff * -1.0j)

    # Generate the list of basis change gates using the p_letters list
    basis_changes = [
        gates.H(_qubit) if _gate == "X" else gates.RX(_qubit, -0.5 * np.pi, trainable=False)
        for _qubit, _gate in p_letters
        if _gate != "Z"
    ]

    # Build the circuit
    circuit = models.Circuit(n_qubits)
    # 1. Change to X/Y where necessary
    circuit.add(_gate for _gate in basis_changes)
    # 2. Add CNOTs to all pairs of qubits in p_letters, starting from the last letter
    circuit.add(
        gates.CNOT(_qubit1, _qubit2) for (_qubit1, _g1), (_qubit2, _g2) in zip(p_letters[::-1], p_letters[::-1][1:])
    )
    # 3. Add RZ gate to last element of p_letters
    circuit.add(gates.RZ(p_letters[0][0], coeff * theta))
    # 4. Add CNOTs to all pairs of qubits in p_letters
    circuit.add(gates.CNOT(_qubit2, _qubit1) for (_qubit1, _g1), (_qubit2, _g2) in zip(p_letters, p_letters[1:]))
    # 3. Change back to the Z basis
    # .dagger() doesn't keep trainable=False, so need to use a for loop
    # circuit.add(_gate.dagger() for _gate in basis_changes)
    for _gate in basis_changes:
        gate = _gate.dagger()
        gate.trainable = False
        circuit.add(gate)
    return circuit, coeff


def ucc_circuit(n_qubits, excitation, theta=0.0, trotter_steps=1, ferm_qubit_map=None, coeffs=None):
    r"""
    Circuit corresponding to the unitary coupled-cluster ansatz for a single excitation

    Args:
        n_qubits: Number of qubits in the quantum circuit
        excitation: Iterable of orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta: UCC parameter. Defaults to 0.0
        trotter_steps: Number of Trotter steps; i.e. number of times the UCC ansatz is applied with \theta = \theta / trotter_steps
        ferm_qubit_map: Fermion-to-qubit transformation. Default is Jordan-Wigner (``jw``).
        coeffs: List to hold the coefficients for the rotation parameter in each Pauli string.
            May be useful in running the VQE. WARNING: Will be modified in this function

    Returns:
        Qibo ``Circuit`` corresponding to a single UCC excitation
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
    circuit = models.Circuit(n_qubits)
    for _i in range(trotter_steps):
        # Use the get_operators() generator to get the list of excitation operators
        for pauli_string in qubit_ucc_operator.get_operators():
            _circuit, coeff = expi_pauli(n_qubits, theta / trotter_steps, pauli_string)
            circuit += _circuit
            if isinstance(coeffs, list):
                coeffs.append(coeff)

    return circuit


# Utility functions


def mp2_amplitude(orbitals, orbital_energies, tei):
    """
    Calculate the MP2 guess amplitudes to be used in the UCC doubles ansatz
        In SO basis: t_{ij}^{ab} = (g_{ijab} - g_{ijba}) / (e_i + e_j - e_a - e_b)

    Args:
        orbitals: list of spin-orbitals representing a double excitation, must have exactly
            4 elements
        orbital_energies: eigenvalues of the Fock operator, i.e. orbital energies
        tei: Two-electron integrals in MO basis and second quantization notation
    """
    # Checks orbitals
    assert len(orbitals) == 4, f"{orbitals} must have only 4 orbitals for a double excitation"
    # Convert orbital indices to be in MO basis
    mo_orbitals = [_orb // 2 for _orb in orbitals]

    # Numerator: g_ijab - g_ijba
    g_ijab = (
        tei[tuple(mo_orbitals)]  # Can index directly using the MO TEIs
        if (orbitals[0] + orbitals[3]) % 2 == 0 and (orbitals[1] + orbitals[2]) % 2 == 0
        else 0.0
    )
    g_ijba = (
        tei[tuple(mo_orbitals[:2] + mo_orbitals[2:][::-1])]  # Reverse last two terms
        if (orbitals[0] + orbitals[2]) % 2 == 0 and (orbitals[1] + orbitals[3]) % 2 == 0
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

    from itertools import combinations

    # Generate all possible excitations first
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
    # Default sorting is OK for single excitations
    sorting_fn = lambda x: sum((order + 1 - _i) * abs(x[2 * _i + 1] // 2 - x[2 * _i] // 2) for _i in range(0, order))

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
                    # 'Move' the first entry of same_mo_index from copy_excitations to result
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
        # Remove the first entry from the sorted list of remaining excitations and add it to result
        prev = copy_excitations.pop(0)
        result.append(prev)
    return result


# def ucc_ansatz(molecule, excitation_level=None, excitations=None, thetas=None, trotter_steps=1, ferm_qubit_map=None):
#     """
#     Build a circuit corresponding to the UCC ansatz with multiple excitations for a given Molecule.
#         If no excitations are given, it defaults to returning the full UCCSD circuit ansatz for the
#         Molecule.
#
#     Args:
#         molecule: Molecule of interest.
#         excitation_level: Include excitations up to how many electrons, i.e. ("S", "D", "T", "Q")
#             Ignored if `excitations` argument is given. Default is "D", i.e. double excitations
#         excitations: List of excitations (e.g. `[[0, 1, 2, 3], [0, 1, 4, 5]]`) used to build the
#             UCC circuit. Overrides the `excitation_level` argument
#         thetas: Parameters for the excitations. Defaults to an array of zeros if not given
#         trotter_steps: number of Trotter steps; i.e. number of times the UCC ansatz is applied with
#             theta=theta/trotter_steps. Default is 1
#         ferm_qubit_map: fermion-to-qubit transformation. Default is Jordan-Wigner ("jw")
#
#     Returns:
#         circuit: Qibo Circuit corresponding to the UCC ansatz
#     """
#     # Get the number of electrons and virtual orbitals from the molecule argument
#     n_elec = sum(molecule.nelec) if molecule.n_active_e is None else molecule.n_active_e
#     n_orbs = molecule.nso if molecule.n_active_orbs is None else molecule.n_active_orbs
#
#     # Define the excitation level to be used if no excitations given
#     if excitations is None:
#         excitation_levels = ("S", "D", "T", "Q")
#         if excitation_level is None:
#             excitation_level = "D"
#         else:
#             # Check validity of input
#             assert len(excitation_level) == 1 and excitation_level.upper() in excitation_levels
#             # Note: Probably don't be too ambitious and try to do 'T'/'Q' at the moment...
#             if excitation_level.upper() in ("T", "Q"):
#                 raise NotImplementedError("Cannot handle triple and quadruple excitations!")
#         # Get the (largest) order of excitation to use
#         excitation_order = excitation_levels.index(excitation_level.upper()) + 1
#
#         # Generate and sort all the possible excitations
#         excitations = []
#         for order in range(excitation_order, 0, -1):  # Reversed to get higher excitations first
#             excitations += sort_excitations(generate_excitations(order, range(0, n_elec), range(n_elec, n_orbs)))
#     else:
#         # Some checks to ensure the given excitations are valid
#         assert all(len(_ex) % 2 == 0 for _ex in excitations), "Excitation with an odd number of elements found!"
#
#     # Check if thetas argument given, define to be all zeros if not
#     # TODO: Unsure if want to use MP2 guess amplitudes for the doubles? Some say good, some say bad
#     # Number of circuit parameters: S->2, D->8, (T/Q->32/128; Not sure?)
#     n_parameters = 2 * len([_ex for _ex in excitations if len(_ex) == 2])  # Singles
#     n_parameters += 8 * len([_ex for _ex in excitations if len(_ex) == 4])  # Doubles
#     if thetas is None:
#         thetas = np.zeros(n_parameters)
#     else:
#         # Check that number of circuit variables (i.e. thetas) matches the number of circuit parameters
#         assert len(thetas) == n_parameters, "Number of input parameters doesn't match the number of circuit parameters!"
#
#     # Build the circuit
#     circuit = models.Circuit(n_orbs)
#     for excitation, theta in zip(excitations, thetas):
#         # coeffs = []
#         circuit += ucc_circuit(
#             n_orbs, excitation, theta, trotter_steps=trotter_steps, ferm_qubit_map=ferm_qubit_map  # , coeffs=coeffs)
#         )
#         # if isinstance(all_coeffs, list):
#         #     all_coeffs.append(np.array(coeffs))
#
#     return circuit
