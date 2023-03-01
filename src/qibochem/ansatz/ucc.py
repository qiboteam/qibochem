"""
TODO: Module docstring

Old: Circuit representing a Hartree-Fock reference wave function
"""

import numpy as np

from qibo import models, gates


def expi_pauli(circuit, theta, pauli_string):
    """
    Apply exp(i*theta*pauli_string) to a circuit

    Args:
        circuit: Qibo circuit object
        theta: parameter
        pauli_string: OpenFermion QubitOperator object
    """
    # Initialise some variables first
    # Unpack the dictionary from pauli_string.terms.items() into:
    # (tuple of Pauli letters), coefficient of Pauli word
    [(p_letters, _coeff)] = pauli_string.terms.items()

    # _coeff is an imaginary number, i.e. exp(i\theta something)
    # Convert to the real coefficient for multiplying theta
    coeff = np.real(_coeff * -1.j)
    # Technically, should have an additional -2 coefficient to \theta because RZ(\theta) = e^{-i/2 \theta Z}
    # But adding it kills my parameter shift rule function:
    # Roughly, f(2\theta + np.pi) - f(2\theta -np.pi) = 0 for all theta, bad!
    # The additional -2 coefficient to \theta is added because RZ(-2\theta) = e^{i\theta Z}
    # Doesn't really affect anything, but just to be more technically exact
    # coeff = -2.*np.real(_coeff * -1.j)

    # Generate the list of basis change gates using the p_letters list
    basis_changes = [gates.H(_qubit) if _gate == 'X' else gates.RX(_qubit, -0.5*np.pi)
                     for _qubit, _gate in p_letters
                     if _gate != "Z"
                    ]

    # Apply the gates corresponding to exp(i ...) to circuit
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
    circuit.add(_gate.dagger() for _gate in basis_changes)

    return circuit


def ucc_ansatz(circuit, theta, orbitals, trotter_steps=1, mapping=None):
    '''
    Adds gates corresponding to the unitary coupled-cluster double excitation operator to a circuit

    Args:
        circuit: circuit object
        theta: parameter
        orbitals: list of orbitals, must have even number of elements
        trotter_steps: number of Trotter steps
            -> i.e. number of times the UCC ansatz is applied with theta=theta/trotter_steps
        mapping: fermion->qubit transformation. Default is Jordan-Wigner (jw)
    '''
    # Check size of orbitals input
    n_orbitals = len(orbitals)
    assert n_orbitals % 2 == 0, f"{orbitals} must have an even number of items!"
    # Sort orbitals
    sorted_orbitals = sorted(orbitals)

    # Define default mapping
    if mapping is None:
        mapping = 'jw'

    # Define the UCC excitation operator corresponding to the given list of orbitals
    fermion_operator_str = ("{}{}".format((n_orbitals//2)*"{}^ ", (n_orbitals//2)*"{} ")
                            .format(*sorted_orbitals[::-1]))
    fermion_operator = openfermion.FermionOperator(fermion_operator_str)
    ucc_operator = (fermion_operator - openfermion.hermitian_conjugated(fermion_operator))

    # Map the FermionOperator to a QubitOperator
    if mapping == 'jw':
        qubit_ucc_operator = openfermion.jordan_wigner(ucc_operator)
    # ZC note: Just putting in ATM, not using
    elif mapping == 'bk':
        qubit_ucc_operator = openfermion.bravyi_kitaev(ucc_operator)
    else:
        print('Error: unknown mapping, returning original circuit!')
        return circuit

    # Apply the qubit_ucc_operator 'trotter_steps' times:
    assert trotter_steps > 0, f"{trotter_steps} must be > 0!"
    for _i in range(trotter_steps):
        # Use the get_operators() generator to get the list of excitation operators
        for pauli_string in qubit_ucc_operator.get_operators():
            circuit = expi_pauli(circuit, theta/trotter_steps, pauli_string)

    return circuit


def ucc_circuit(circuit, thetas, excitations, trotter_params, mapping=None):
    """
    Helper function to apply the UCC ansatz w.r.t. a list of excitations to a circuit

    Args:
        circuit: Circuit to apply the UCC ansatz (plural: ansatzes?)
        thetas: Parameters for the UCC excitations
        excitations: List of excitations to be applied
        trotter_params: Number of Trotter steps to be applied for each excitation
        mapping: Fermion->Qubit mapping, e.g. 'jw'
    """
    # Check that the dimensions of thetas, excitations, and trotter_parms match
    assert len(thetas) == len(excitations), (f"No. of input parameters: {len(thetas)} != "
                                             f"{len(excitations)}: No. of excitations!")
    assert len(excitations) == len(trotter_params), (f"No. of excitations: {len(excitations)} != "
                                                     f"{len(trotter_params)}: No. of Trotter parameters!")
    # Add UCC gates for each excitation
    for theta, excitation, t_steps in zip(thetas, excitations, trotter_params):
        circuit = ucc_ansatz(circuit, theta, excitation, trotter_steps=t_steps, mapping=mapping)

    return circuit
