"""
Helper functions for obtaining and transforming the molecular Hamiltonian
"""

import numpy as np
import openfermion
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian
from sympy import Add, Mul


def _fermionic_hamiltonian(
    oei: np.ndarray, tei: np.ndarray, constant: float
) -> openfermion.InteractionOperator:
    """
    Build molecular Hamiltonian using the 1-/2- electron integrals

    Args:
        oei (np.ndarray):
            1-electron integrals in the MO basis
        tei (np.ndarray):
            2-electron integrals in 2ndQ notation and MO basis
        constant (float):
            Nuclear-nuclear repulsion, and inactive Fock energy if HF embedding used

    Returns:
        openfermion.InteractionOperator: Molecular Hamiltonian
    """
    oei_so, tei_so = openfermion.ops.representations.get_tensors_from_integrals(
        oei, tei
    )
    # tei_so already multiplied by 0.5, no need to include in InteractionOperator
    return openfermion.InteractionOperator(constant, oei_so, tei_so)


def _qubit_hamiltonian(
    fermion_hamiltonian: openfermion.InteractionOperator, ferm_qubit_map: str
) -> openfermion.QubitOperator:
    """
    Converts the molecular Hamiltonian to a QubitOperator

    Args:
        fermion_hamiltonian (openfermion.InteractionOperator):
             Molecular Hamiltonian
        ferm_qubit_map (str):
            Which Fermion->Qubit mapping to use

    Returns:
        openfermion.QubitOperator: Molecular Hamiltonian
    """
    # Map the fermionic molecular Hamiltonian to a QubitHamiltonian
    if ferm_qubit_map == "jw":
        q_hamiltonian = openfermion.jordan_wigner(fermion_hamiltonian)
    elif ferm_qubit_map == "bk":
        q_hamiltonian = openfermion.bravyi_kitaev(fermion_hamiltonian)
    else:
        error = f"Unknown fermion->qubit mapping: {ferm_qubit_map}"
        raise KeyError(error)
    q_hamiltonian.compress()  # Remove terms with v. small coefficients
    return q_hamiltonian


def _qubit_to_symbolic_hamiltonian(
    q_hamiltonian: openfermion.QubitOperator,
) -> SymbolicHamiltonian:
    """
    Converts a OpenFermion QubitOperator to a Qibo SymbolicHamiltonian

    Args:
        q_hamiltonian (openfermion.QubitOperator): Molecular Hamiltonian

    Returns:
        SymbolicHamiltonian: Molecular Hamiltonian
    """
    # Use sympy operations without simplifying the whole expression for every term
    symbolic_expr = Add(
        *(
            Mul(
                coeff,
                *(
                    getattr(symbols, pauli_op)(qubit)
                    for qubit, pauli_op in pauli_string
                ),
                evaluate=False,
            )
            for pauli_string, coeff in q_hamiltonian.terms.items()
        ),
        evaluate=False,
    )
    return SymbolicHamiltonian(symbolic_expr)
