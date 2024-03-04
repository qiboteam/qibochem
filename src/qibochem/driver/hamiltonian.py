"""
Functions for obtaining and transforming the molecular Hamiltonian
"""

from functools import reduce

import openfermion
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian


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


def qubit_to_symbolic_hamiltonian(q_hamiltonian):
    """
    Converts a OpenFermion QubitOperator to a Qibo SymbolicHamiltonian

    Args:
        q_hamiltonian: QubitOperator

    Returns:
        qibo.hamiltonians.SymbolicHamiltonian
    """
    symbolic_ham = sum(
        reduce(lambda x, y: x * y, (getattr(symbols, pauli_op)(qubit) for qubit, pauli_op in pauli_string), coeff)
        # Sums over each individual Pauli string in the QubitOperator
        for operator in q_hamiltonian.get_operators()
        # .terms gives one operator as a single-item dictionary, e.g. {((1: "X"), (2: "Y")): 0.33}
        for pauli_string, coeff in operator.terms.items()
    )
    return SymbolicHamiltonian(symbolic_ham)
