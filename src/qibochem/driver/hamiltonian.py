"""
Functions for obtaining and transforming the molecular Hamiltonian
"""

import openfermion
from qibo import hamiltonians
from qibo.symbols import X, Y, Z


def fermionic_hamiltonian(oei, tei, constant):
    """
    Build molecular Hamiltonian as an InteractionOperator using the 1-/2- electron integrals

    Args:
        oei: 1-electron integrals in the MO basis
        tei: 2-electron integrals in 2ndQ notation and MO basis
        constant: Nuclear-nuclear repulsion, and inactive Fock energy if HF embedding used

    Returns:
        Molecular Hamiltonian as an InteractionOperator
    """
    oei_so, tei_so = openfermion.ops.representations.get_tensors_from_integrals(oei, tei)
    # tei_so already multiplied by 0.5, no need to include in InteractionOperator
    return openfermion.InteractionOperator(constant, oei_so, tei_so)


def qubit_hamiltonian(fermion_hamiltonian, ferm_qubit_map):
    """
    Converts the molecular Hamiltonian to a QubitOperator

    Args:
        fermion_hamiltonian: Molecular Hamiltonian as a InteractionOperator/FermionOperator
        ferm_qubit_map: Which Fermion->Qubit mapping to use

    Returns:
        qubit_operator : Molecular Hamiltonian as a QubitOperator
    """
    # Map the fermionic molecular Hamiltonian to a QubitHamiltonian
    if ferm_qubit_map == "jw":
        q_hamiltonian = openfermion.jordan_wigner(fermion_hamiltonian)
    elif ferm_qubit_map == "bk":
        q_hamiltonian = openfermion.bravyi_kitaev(fermion_hamiltonian)
    else:
        raise KeyError("Unknown fermion->qubit mapping!")
    q_hamiltonian.compress()  # Remove terms with v. small coefficients
    return q_hamiltonian


def parse_pauli_string(pauli_string, coeff):
    """
    Helper function: Converts a single Pauli string to a Qibo Symbol

    Args:
        pauli_string (tuple of tuples): Indicate what gates to apply onto which qubit
            e.g. ((0, 'Z'), (2, 'Z'))
        coeff (float): Coefficient of the Pauli string

    Returns:
        qibo.symbols.Symbol for a single Pauli string, e.g. -0.04*X0*X1*Y2*Y3
    """
    # Dictionary for converting
    xyz_to_symbol = {"X": X, "Y": Y, "Z": Z}
    # Check that pauli_string is non-empty
    if pauli_string:
        # pauli_string format: ((0, 'Y'), (1, 'Y'), (3, 'X'))
        qibo_pauli_string = 1.0
        for p_letter in pauli_string:
            qibo_pauli_string *= xyz_to_symbol[p_letter[1]](p_letter[0])
        # Include coefficient after all symbols
        qibo_pauli_string = coeff * qibo_pauli_string
    else:
        # Empty word, i.e. constant term in Hamiltonian
        qibo_pauli_string = coeff
    return qibo_pauli_string


def symbolic_hamiltonian(q_hamiltonian):
    """
    Converts a OpenFermion QubitOperator to a Qibo SymbolicHamiltonian

    Args:
        q_hamiltonian: QubitOperator

    Returns:
        qibo.hamiltonians.SymbolicHamiltonian
    """
    # Sums over each individual Pauli string in the QubitOperator
    symbolic_ham = sum(
        parse_pauli_string(pauli_string, coeff)
        # Iterate over all operators
        for operator in q_hamiltonian.get_operators()
        # .terms gives one operator as a dictionary with one entry
        for pauli_string, coeff in operator.terms.items()
    )

    # Map the QubitHamiltonian to a Qibo SymbolicHamiltonian and return it
    return hamiltonians.SymbolicHamiltonian(symbolic_ham)
