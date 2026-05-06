"""
Circuit ansatzes for chemistry
"""

import numpy as np
from qibo import Circuit
from qibo.config import raise_error
from qibo.models.encodings import comp_basis_encoder

from qibochem.ansatz._ansatz import _bk_matrix


def hf_circuit(nqubits: int, nelectrons: int, ferm_qubit_map: str | None = None, **kwargs) -> Circuit:
    """Circuit to prepare a Hartree-Fock state

    Args:
        nqubits (int): Number of qubits in the quantum circuit
        nelectrons (int): Number of electrons in the molecular system
        ferm_qubit_map (str | None, optional): Fermion to qubit map. Must be either Jordan-Wigner (``"jw"``) or Brayvi-Kitaev
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
        raise_error(KeyError, "Fermon-to-qubit mapping must be either 'jw' or 'bk'")

    # Occupation number of SOs
    mapped_occ_n = None
    occ_n = np.concatenate((np.ones(nelectrons, dtype=np.int8), np.zeros(nqubits - nelectrons, dtype=np.int8)))
    if ferm_qubit_map == "jw":
        mapped_occ_n = occ_n
    elif ferm_qubit_map == "bk":
        mapped_occ_n = (_bk_matrix(nqubits) @ occ_n) % 2
    # Convert the array to a list, then build/return the final circuit
    return comp_basis_encoder(mapped_occ_n.tolist(), nqubits=nqubits, **kwargs)
