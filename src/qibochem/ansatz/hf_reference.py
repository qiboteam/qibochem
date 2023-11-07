"""
Circuit representing a Hartree-Fock reference wave function
"""

import numpy as np
from qibo import gates, models


# Helper functions for the Brayvi-Kitaev mapping
def bk_matrix_power2(n: int):
    """Build the Brayvi-Kitaev matrix of dimension n (= power of 2) recursively

    Args:
        n: Size of BK matrix
    """
    assert n > 0, "Dimension of BK matrix must be at least 1"
    # Base case
    if n == 1:
        return np.ones((1, 1), dtype=np.int8)
    # Recursive definition
    smaller_bk_matrix = bk_matrix_power2(n - 1)
    top_right = np.zeros((2 ** (n - 2), 2 ** (n - 2)), dtype=np.int8)
    top_half = np.concatenate((smaller_bk_matrix, top_right), axis=1)

    bottom_left = np.concatenate(
        (np.zeros(((2 ** (n - 2)) - 1, 2 ** (n - 2)), dtype=np.int8), np.ones((1, 2 ** (n - 2)), dtype=np.int8)), axis=0
    )
    bottom_half = np.concatenate((bottom_left, smaller_bk_matrix), axis=1)
    # Combine top and bottom half
    return np.concatenate((top_half, bottom_half), axis=0)


def bk_matrix(n: int):
    """Exact Brayvi-Kitaev matrix of size n, obtained by slicing a larger BK matrix
        with dimension 2**m > n

    TODO: Update the occupation number vector using the update, parity, and flip set instead?
        Not sure if necessary; i.e. size of BK matrix probably not comparable to the memory needed
        for a classical simulation?

    Args:
        n: Size of BK matrix
    """
    assert n > 0, "Dimension of the Brayvi-Kitaev matrix must be at least 1"
    # Build bk_matrix_power2(m), where 2**m > n
    min_bk_size = int(np.ceil(np.log2(n))) + 1
    min_bk_matrix = bk_matrix_power2(min_bk_size)
    # Then use array slicing to get the actual BK matrix
    return min_bk_matrix[:n, :n]


# Main function
def hf_circuit(n_qubits, n_electrons, ferm_qubit_map=None):
    """Circuit to prepare a Hartree-Fock state

    Args:
        n_qubits: Number of qubits in the quantum circuit
        n_electrons: Number of electrons in the molecular system
        ferm_qubit_map: Fermion to qubit map. Must be either Jordan-Wigner (``jw``) or Brayvi-Kitaev (``bk``). Default value is ``jw``.

    Returns:
        Qibo ``Circuit`` initialized in a HF reference state
    """
    # Which fermion-to-qubit map to use
    if ferm_qubit_map is None:
        ferm_qubit_map = "jw"
    if ferm_qubit_map not in ("jw", "bk"):
        raise KeyError("Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Occupation number of SOs
    occ_n = np.concatenate((np.ones(n_electrons), np.zeros(n_qubits - n_electrons)))
    if ferm_qubit_map == "jw":
        mapped_occ_n = occ_n
    elif ferm_qubit_map == "bk":
        mapped_occ_n = (bk_matrix(n_qubits) @ occ_n) % 2
    # Convert to ints
    mapped_occ_n = mapped_occ_n.astype(int)
    # Build the circuit using the mapped vector
    circuit = models.Circuit(n_qubits)
    circuit.add(gates.X(int(_i)) for _i, _mo in enumerate(mapped_occ_n) if _mo == 1)
    return circuit
