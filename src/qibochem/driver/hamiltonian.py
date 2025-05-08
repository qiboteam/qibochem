"""
Helper functions for obtaining and transforming the molecular Hamiltonian
"""

from functools import reduce

import openfermion
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian


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
    symbolic_ham = sum(
        reduce(lambda x, y: x * y, (getattr(symbols, pauli_op)(qubit) for qubit, pauli_op in pauli_string), coeff)
        # Sums over each individual Pauli string in the QubitOperator
        for operator in q_hamiltonian.get_operators()
        # .terms gives one operator as a single-item dictionary, e.g. {((1: "X"), (2: "Y")): 0.33}
        for pauli_string, coeff in operator.terms.items()
    )
    return SymbolicHamiltonian(symbolic_ham)

# --- Folded Hamiltonian utilities for excited state VQE ---
def _symbolic_identity(nqubits):
    """
    Returns the symbolic identity operator for nqubits as a sympy expression using Qibo symbols.
    Args:
        nqubits (int): Number of qubits
    Returns:
        sympy expression representing the identity operator on nqubits
    """
    from qibo.symbols import I
    op = 1
    for q in range(nqubits):
        op *= I(q)
    return op


def build_folded_hamiltonian(q_hamiltonian, lambda_shift):
    """
    Constructs the folded Hamiltonian (H - lambda*I)^2 as a Qibo SymbolicHamiltonian.

    Args:
        q_hamiltonian: Qibo SymbolicHamiltonian or OpenFermion QubitOperator (Hamiltonian H)
        lambda_shift: Scalar value (ground state energy or other shift)

    Returns:
        folded_hamiltonian: Qibo SymbolicHamiltonian representing (H - lambda*I)^2
    """
    if not isinstance(q_hamiltonian, SymbolicHamiltonian):
        q_hamiltonian = _qubit_to_symbolic_hamiltonian(q_hamiltonian)
    nqubits = q_hamiltonian.nqubits
    identity_expr = _symbolic_identity(nqubits)
    identity = SymbolicHamiltonian(identity_expr, nqubits)
    shifted_h = q_hamiltonian - lambda_shift * identity
    folded_h = shifted_h @ shifted_h
    return folded_h