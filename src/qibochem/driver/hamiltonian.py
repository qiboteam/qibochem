"""
Helper functions for obtaining and transforming the molecular Hamiltonian
"""

import openfermion
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian
from sympy import Add, Mul


def _fermionic_hamiltonian(oei, tei, constant):
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


def _qubit_hamiltonian(fermion_hamiltonian, ferm_qubit_map):
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


def _qubit_to_symbolic_hamiltonian(q_hamiltonian):
    """
    Converts a OpenFermion QubitOperator to a Qibo SymbolicHamiltonian

    Args:
        q_hamiltonian: QubitOperator

    Returns:
        qibo.hamiltonians.SymbolicHamiltonian
    """
    # Use sympy operations without evaluating (i.e. simplifying) the whole expression for every term
    pauli_terms = [
        Mul(coeff, *(getattr(symbols, pauli_op)(qubit) for qubit, pauli_op in pauli_string), evaluate=False)
        for pauli_string, coeff in q_hamiltonian.terms.items()
    ]
    symbolic_ham = Add(*pauli_terms, evaluate=False)
    return SymbolicHamiltonian(symbolic_ham)
