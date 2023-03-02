"""
Circuit representing the Unitary Coupled Cluster ansatz in quantum chemistry
"""

import numpy as np
import openfermion

from qibo import models, gates


def expi_pauli(n_qubits, theta, pauli_string):
    """
    Build circuit representing exp(i*theta*pauli_string) to a circuit

    Args:
        n_qubits: No. of qubits in the quantum circuit
        theta: parameter
        pauli_string: OpenFermion QubitOperator object, e.g. X_0 Y_1 Z_2 X_4
    """
    # Unpack the dictionary from pauli_string.terms.items() into
    # a (tuple of Pauli letters) and the coefficient of the pauli_string
    [(p_letters, _coeff)] = pauli_string.terms.items()

    # _coeff is an imaginary number, i.e. exp(i\theta something)
    # Convert to the real coefficient for multiplying theta
    coeff = -2.*np.real(_coeff * -1.j)

    # Generate the list of basis change gates using the p_letters list
    basis_changes = [gates.H(_qubit) if _gate == 'X' else gates.RX(_qubit, -0.5*np.pi)
                     for _qubit, _gate in p_letters
                     if _gate != "Z"
                    ]

    # Apply the gates corresponding to exp(i ...) to a quantum circuit
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
    circuit.add(_gate.dagger() for _gate in basis_changes)

    return circuit


def ucc_circuit(n_qubits, theta, orbitals, trotter_steps=1, mapping=None):
    '''
    Build circuit corresponding to the full unitary coupled-cluster ansatz

    Args:
        n_qubits: No. of qubits in the quantum circuit
        theta: parameter
        orbitals: list of orbitals, must have even number of elements
            e.g. [0, 1, 2, 3] represents the excitation of electrons in orbitals (0, 1) to (2, 3)
        trotter_steps: number of Trotter steps
            -> i.e. number of times the UCC ansatz is applied with theta=theta/trotter_steps
        mapping: fermion->qubit transformation. Default is Jordan-Wigner (jw)
    '''
    # Check size of orbitals input
    n_orbitals = len(orbitals)
    assert n_orbitals % 2 == 0, f"{orbitals} must have an even number of items"
    # Reverse sort orbitals to get largest-->smallest
    sorted_orbitals = sorted(orbitals, reverse=True)

    # Define default mapping
    if mapping is None:
        mapping = 'jw'

    # Define the UCC excitation operator corresponding to the given list of orbitals
    fermion_op_str_template = f"{(n_orbitals//2)*'{}^ '}{(n_orbitals//2)*'{} '}"
    fermion_operator_str = fermion_op_str_template.format(*sorted_orbitals)
    # Build the FermionOperator and make it unitary
    fermion_operator = openfermion.FermionOperator(fermion_operator_str)
    ucc_operator = (fermion_operator - openfermion.hermitian_conjugated(fermion_operator))

    # Map the FermionOperator to a QubitOperator
    if mapping == 'jw':
        qubit_ucc_operator = openfermion.jordan_wigner(ucc_operator)
    # ZC note: Just putting in ATM, not using
    elif mapping == 'bk':
        qubit_ucc_operator = openfermion.bravyi_kitaev(ucc_operator)
    else:
        raise KeyError("Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Apply the qubit_ucc_operator 'trotter_steps' times:
    assert trotter_steps > 0, f"{trotter_steps} must be > 0!"
    circuit = models.Circuit(n_qubits)
    for _i in range(trotter_steps):
        # Use the get_operators() generator to get the list of excitation operators
        for pauli_string in qubit_ucc_operator.get_operators():
            circuit += expi_pauli(n_qubits, theta/trotter_steps, pauli_string)

    return circuit
