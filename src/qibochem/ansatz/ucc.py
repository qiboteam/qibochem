"""
Circuit representing the Unitary Coupled Cluster ansatz in quantum chemistry
"""

import numpy as np
import openfermion

from qibo import models, gates


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
    mo_orbitals = [_orb//2 for _orb in orbitals]

    # Numerator: g_ijab - g_ijba
    g_ijab = (tei[tuple(mo_orbitals)] # Can index directly using the MO TEIs
              if (orbitals[0] + orbitals[3]) % 2 == 0 and (orbitals[1] + orbitals[2]) % 2 == 0
              else 0.0
             )
    g_ijba = (tei[tuple(mo_orbitals[:2] + mo_orbitals[2:][::-1])] # Reverse last two terms
              if (orbitals[0] + orbitals[2]) % 2 == 0 and (orbitals[1] + orbitals[3]) % 2 == 0
              else 0.0
             )
    numerator = g_ijab - g_ijba
    # Denominator is directly from the orbital energies
    denominator = sum(orbital_energies[mo_orbitals[:2]]) - sum(orbital_energies[mo_orbitals[2:]])
    return numerator / denominator


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
    coeff = -2.*np.real(_coeff * -1.j)

    # Generate the list of basis change gates using the p_letters list
    basis_changes = [gates.H(_qubit) if _gate == 'X'
                     else gates.RX(_qubit, -0.5*np.pi, trainable=False)
                     for _qubit, _gate in p_letters
                     if _gate != "Z"
                    ]

    # Build the circuit
    circuit = models.Circuit(n_qubits)
    # 1. Change to X/Y where necessary
    circuit.add(_gate for _gate in basis_changes)
    # 2. Add CNOTs to all pairs of qubits in p_letters, starting from the last letter
    circuit.add(gates.CNOT(_qubit1, _qubit2)
                for (_qubit1, _g1), (_qubit2, _g2) in zip(p_letters[::-1], p_letters[::-1][1:]))
    # 3. Add RZ gate to last element of p_letters
    circuit.add(gates.RZ(p_letters[0][0], coeff*theta))
    # 4. Add CNOTs to all pairs of qubits in p_letters
    circuit.add(gates.CNOT(_qubit2, _qubit1)
                for (_qubit1, _g1), (_qubit2, _g2) in zip(p_letters, p_letters[1:]))
    # 3. Change back to the Z basis
    # .dagger() doesn't keep trainable=False, so need to use a for loop
    # circuit.add(_gate.dagger() for _gate in basis_changes)
    for _gate in basis_changes:
        gate = _gate.dagger()
        gate.trainable = False
        circuit.add(gate)
    return circuit, coeff


def ucc_circuit(n_qubits, theta, orbitals, trotter_steps=1, ferm_qubit_map=None, coeffs=None):
    '''
    Build circuit corresponding to the full unitary coupled-cluster ansatz

    Args:
        n_qubits: No. of qubits in the quantum circuit
        theta: parameter
        orbitals: list of orbitals, must have even number of elements
            e.g. [0, 1, 2, 3] represents the excitation of electrons in orbitals (0, 1) to (2, 3)
        trotter_steps: number of Trotter steps
            -> i.e. number of times the UCC ansatz is applied with theta=theta/trotter_steps
        ferm_qubit_map: fermion->qubit transformation. Default is Jordan-Wigner (jw)
        coeffs: List to hold the coefficients for the rotation parameter in each Pauli string.
            May be useful in running the VQE. WARNING: Will be modified in this function
    '''
    # Check size of orbitals input
    n_orbitals = len(orbitals)
    assert n_orbitals % 2 == 0, f"{orbitals} must have an even number of items"
    # Reverse sort orbitals to get largest-->smallest
    sorted_orbitals = sorted(orbitals, reverse=True)

    # Define default mapping
    if ferm_qubit_map is None:
        ferm_qubit_map = 'jw'

    # Define the UCC excitation operator corresponding to the given list of orbitals
    fermion_op_str_template = f"{(n_orbitals//2)*'{}^ '}{(n_orbitals//2)*'{} '}"
    fermion_operator_str = fermion_op_str_template.format(*sorted_orbitals)
    # Build the FermionOperator and make it unitary
    fermion_operator = openfermion.FermionOperator(fermion_operator_str)
    ucc_operator = (fermion_operator - openfermion.hermitian_conjugated(fermion_operator))

    # Map the FermionOperator to a QubitOperator
    if ferm_qubit_map == 'jw':
        qubit_ucc_operator = openfermion.jordan_wigner(ucc_operator)
    elif ferm_qubit_map == 'bk':
        qubit_ucc_operator = openfermion.bravyi_kitaev(ucc_operator)
    else:
        raise KeyError("Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Apply the qubit_ucc_operator 'trotter_steps' times:
    assert trotter_steps > 0, f"{trotter_steps} must be > 0!"
    circuit = models.Circuit(n_qubits)
    for _i in range(trotter_steps):
        # Use the get_operators() generator to get the list of excitation operators
        for pauli_string in qubit_ucc_operator.get_operators():
            _circuit, coeff = expi_pauli(n_qubits, theta/trotter_steps, pauli_string)
            circuit += _circuit
            if isinstance(coeffs, list):
                coeffs.append(coeff)

    return circuit


# Utility functions for running a UCCSD calculation

def generate_excitations(order, excite_from, excite_to, conserve_spin=True):
    """
    Generate all possible excitations between two lists of integers

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
    return [[*_from, *_to]
            for _from in combinations(excite_from, order)
            for _to in combinations(excite_to, order)
            if (not conserve_spin
                or (sum([*_from, *_to]) % 2 == 0
                    and sum([_i % 2 for _i in _from]) == sum([_i % 2 for _i in _to]))
               )
           ]


def sort_excitations(excitations):
    """
    TODO: Docstring

    Sorts the list of excitations according to some common-sense and empirical rules

    Sorting order:
    1. (For double excitations only) All paired excitations between the same MOs first
    2. Pair up excitations between the same MOs, e.g. (0, 2) and (1, 3) 
    3. Then count upwards from smallest MO

    """
    # Check that all excitations are of the same order
    order = len(excitations[0]) // 2
    if order > 2:
        raise NotImplementedError("Can only handle single and double excitations!")

    assert all(len(_ex)//2 == order for _ex in excitations), "Cannot handle excitations of different orders!"
    # TODO: Actually not that difficult to do? Just sort by excitation length and split up?
    # Can probably implement in future...

    # Define variables for the while loop
    copy_excitations = excitations.copy()
    result = []
    prev = None

    # Some comment for my future self
    while copy_excitations:
        if prev is None:
            # Sort the remaining excitations
            copy_excitations = sorted(copy_excitations)
        else:
            # Check to see for excitations involving the same MOs as prev
            _from = prev[:order]
            new_from = [_i+1 if _i % 2 == 0 else _i - 1 for _i in _from]
            _to = prev[order:]
            new_to = [_i+1 if _i % 2 == 0 else _i - 1 for _i in _to]
            new_ex = sorted(new_from + new_to)

            # Any such excitations left in the list?
            if new_ex in copy_excitations:
                index = copy_excitations.index(new_ex)
                result.append(copy_excitations.pop(index))
                prev = None
                continue
            else:
                # No excitations involving the same set of MOs
                # Move on to other MOs, with paired excitations first
                copy_excitations = sorted(copy_excitations,
                                          key=lambda x: abs(x[1]//2 - x[0]//2) + abs(x[3]//2 - x[2]//2)
                                         )
        # Take out the first entry from the sorted list of remaining excitations and add it to result
        prev = copy_excitations.pop(0)
        result.append(prev)
    return result


def uccsd_circuit(n_qubits, n_electrons, trotter_steps=1, ferm_qubit_map=None, all_coeffs=None):
    """
    TODO: Docstring

    Utility function to build the whole UCCSD ansatz for a given molecule

    TODO: Expand to a general number of excitations?

    Args:
        n_qubits = Number of spin-orbitals in the system (== Number of qubits required)

    Returns:
        circuit:
    """
    circuit = models.Circuit(n_qubits)

    for order in range(2, 0, -1): # Reverse order because we want double excitations first
        excitations = sort_excitations(
            generate_excitations(order, range(0, n_electrons), range(n_electrons, n_qubits))
        )
        coeffs = []
        for excitation in excitations:
            circuit += ucc_circuit(n_qubits, 0.0, excitation, trotter_steps=trotter_steps, ferm_qubit_map=ferm_qubit_map, coeffs=coeffs)
            if isinstance(all_coeffs, list):
                all_coeffs.append(np.array(coeffs))

    return circuit
