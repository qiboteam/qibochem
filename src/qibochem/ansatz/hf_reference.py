"""
Circuit representing a Hartree-Fock reference wave function
"""

import numpy as np
from qibo.models.encodings import comp_basis_encoder


# Helper functions for the Brayvi-Kitaev mapping
def _bk_matrix_power2(n: int):
    """Build the Brayvi-Kitaev matrix of dimension n (= power of 2) recursively

    Args:
        n: Size of BK matrix
    """
    assert n > 0, "Dimension of BK matrix must be at least 1"
    # Base case
    if n == 1:
        return np.ones((1, 1), dtype=np.int8)
    # Recursive definition
    smaller_bk_matrix = _bk_matrix_power2(n - 1)
    top_right = np.zeros((2 ** (n - 2), 2 ** (n - 2)), dtype=np.int8)
    top_half = np.concatenate((smaller_bk_matrix, top_right), axis=1)

    bottom_left = np.concatenate(
        (np.zeros(((2 ** (n - 2)) - 1, 2 ** (n - 2)), dtype=np.int8), np.ones((1, 2 ** (n - 2)), dtype=np.int8)), axis=0
    )
    bottom_half = np.concatenate((bottom_left, smaller_bk_matrix), axis=1)
    # Combine top and bottom half
    return np.concatenate((top_half, bottom_half), axis=0)


def _bk_matrix(n: int):
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
    min_bk_matrix = _bk_matrix_power2(min_bk_size)
    # Then use array slicing to get the actual BK matrix
    return min_bk_matrix[:n, :n]


# Main function
def hf_circuit(n_qubits, n_electrons, ferm_qubit_map=None, **kwargs):
    """Circuit to prepare a Hartree-Fock state

    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        n_electrons (int): Number of electrons in the molecular system
        ferm_qubit_map (str, optional): Fermion to qubit map. Must be either Jordan-Wigner (``"jw"``) or Brayvi-Kitaev
            (``"bk"``). Default value is ``"jw"``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object. Details are given in the
            documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit initialized in a HF reference state
    """
    # Which fermion-to-qubit map to use
    if ferm_qubit_map is None:
        ferm_qubit_map = "jw"
    if ferm_qubit_map not in ("jw", "bk"):
        raise KeyError("Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Occupation number of SOs
    mapped_occ_n = None
    occ_n = np.concatenate((np.ones(n_electrons, dtype=np.int8), np.zeros(n_qubits - n_electrons, dtype=np.int8)))
    if ferm_qubit_map == "jw":
        mapped_occ_n = occ_n
    elif ferm_qubit_map == "bk":
        mapped_occ_n = (_bk_matrix(n_qubits) @ occ_n) % 2
    # Convert the array to a list, then build/return the final circuit
    return comp_basis_encoder(mapped_occ_n.tolist(), nqubits=n_qubits, **kwargs)
