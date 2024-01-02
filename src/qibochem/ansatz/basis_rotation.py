"""
Circuit representing an unitary rotation of the molecular (spin-)orbital basis set
"""

import numpy as np
from openfermion.linalg.givens_rotations import givens_decomposition
from qibo import gates, models
from scipy.linalg import expm

# Helper functions


def unitary_rot_matrix(parameters, occ_orbitals, virt_orbitals):
    r"""
    Returns the unitary rotation matrix U = exp(\kappa) mixing the occupied and virtual orbitals,
         where \kappa is an anti-Hermitian matrix

    Args:
        parameters: List/array of rotation parameters. dim = len(occ_orbitals)*len(virt_orbitals)
        occ_orbitals: Iterable of occupied orbitals
        virt_orbitals: Iterable of virtual orbitals
    """
    n_orbitals = len(occ_orbitals) + len(virt_orbitals)

    kappa = np.zeros((n_orbitals, n_orbitals))
    # Get all possible (occ, virt) pairs
    orbital_pairs = [(_o, _v) for _o in occ_orbitals for _v in virt_orbitals]
    # occ/occ and virt/virt orbital mixing doesn't change the energy, so can ignore
    for _i, (_occ, _virt) in enumerate(orbital_pairs):
        kappa[_occ, _virt] = parameters[_i]
        kappa[_virt, _occ] = -parameters[_i]
    return expm(kappa)


def swap_matrices(permutations, n_qubits):
    r"""
    Build matrices that permute (swap) the columns of a given matrix

    Args:
        permutations [List(tuple, ), ]: List of lists of 2-tuples permuting two consecutive numbers
            e.g. [[(0, 1), (2, 3)], [(1, 2)], ...]
        n_qubits: Dimension of matrix (\exp(\kappa) matrix) to be operated on i.e. # of qubits

    Returns:
        List of matrices that carry out \exp(swap) for each pair in permutations
    """
    exp_swaps = []
    _temp = np.array([[-1.0, 1.0], [1.0, -1.0]])
    for pairs in permutations:
        gen = np.zeros((n_qubits, n_qubits))
        for pair in pairs:
            # Add zeros to pad out the (2, 2) matrix to (n_qubits, n_qubits) with 0.
            gen += np.pad(_temp, ((pair[0], n_qubits - pair[1] - 1),), "constant")
        # Take matrix exponential, remove all entries close to zero, and convert to real matrix
        exp_mat = expm(-0.5j * np.pi * gen)
        exp_mat[np.abs(exp_mat) < 1e-10] = 0.0
        exp_mat = np.real_if_close(exp_mat)
        # Add exp_mat to exp_swaps once cleaned up
        exp_swaps.append(exp_mat)
    return exp_swaps


def givens_rotation_parameters(n_qubits, exp_k, n_occ):
    r"""
    Get the parameters of the Givens rotation matrix obtained after decomposing \exp(\kappa)

    Args:
        n_qubits: Number of qubits (i.e. spin-orbitals)
        exp_k: Unitary rotation matrix
        n_occ: Number of occupied orbitals
    """
    # List of 2-tuples to link all qubits via swapping
    swap_pairs = [
        [(_i, _i + 1) for _i in range(_d % 2, n_qubits - 1, 2)]
        # Only 2 possible ways of swapping: [(0, 1), ... ] or [(1, 2), ...]
        for _d in range(2)
    ]
    # Get their corresponding matrices
    swap_unitaries = swap_matrices(swap_pairs, n_qubits)

    # Create a copy of exp_k for rotating
    exp_k_shift = np.copy(exp_k)
    # Apply the column-swapping transformations
    for u_rot in swap_unitaries:
        exp_k_shift = u_rot @ exp_k_shift
    # Only want the n_occ by n_qubits (n_orb) sub-matrix
    qmatrix = exp_k_shift.T[:n_occ, :]

    # Apply Givens decomposition of the submatrix to get a list of individual Givens rotations
    # (orb1, orb2, theta, phi), where orb1 and orb2 are rotated by theta(real) and phi(imag)
    qdecomp, _, _ = givens_decomposition(qmatrix)
    # Lastly, reverse the list of Givens rotations (because U = G_k ... G_2 G_1)
    return list(reversed(qdecomp))


def givens_rotation_gate(n_qubits, orb1, orb2, theta):
    """
    Circuit corresponding to a Givens rotation between two qubits (spin-orbitals)

    Args:
        n_qubits: Number of qubits in the quantum circuit
        orb1, orb2: Orbitals used for the Givens rotation
        theta: Rotation parameter

    Returns:
        circuit: Qibo Circuit object representing a Givens rotation between orb1 and orb2
    """
    # Define 2x2 sqrt(iSwap) matrix
    iswap_mat = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    # Build and add the gates to circuit
    circuit = models.Circuit(n_qubits)
    circuit.add(gates.GeneralizedfSim(orb1, orb2, iswap_mat, 0.0, trainable=False))
    circuit.add(gates.RZ(orb1, -theta))
    circuit.add(gates.RZ(orb2, theta + np.pi))
    circuit.add(gates.GeneralizedfSim(orb1, orb2, iswap_mat, 0.0, trainable=False))
    circuit.add(gates.RZ(orb1, np.pi, trainable=False))
    return circuit


def br_circuit(n_qubits, parameters, n_occ):
    """
    Google's basis rotation circuit, applied between the occupied/virtual orbitals. Forms the exp(kappa) matrix, decomposes
    it into Givens rotations, and sets the circuit parameters based on the Givens rotation decomposition. Note: Supposed
    to be used with the JW fermion-to-qubit mapping

    Args:
        n_qubits: Number of qubits in the quantum circuit
        parameters: Rotation parameters for exp(kappa); Must have (n_occ * n_virt) parameters
        n_occ: Number of occupied orbitals

    Returns:
        Qibo ``Circuit`` corresponding to the basis rotation ansatz between the occupied and virtual orbitals
    """
    assert len(parameters) == (n_occ * (n_qubits - n_occ)), "Need len(parameters) == (n_occ * n_virt)"
    # Unitary rotation matrix \exp(\kappa)
    exp_k = unitary_rot_matrix(parameters, range(n_occ), range(n_occ, n_qubits))
    # List of Givens rotation parameters
    g_rotation_parameters = givens_rotation_parameters(n_qubits, exp_k, n_occ)

    # Start building the circuit
    circuit = models.Circuit(n_qubits)
    # Add the Givens rotation circuits
    for g_rot_parameter in g_rotation_parameters:
        for orb1, orb2, theta, _phi in g_rot_parameter:
            assert np.allclose(_phi, 0.0), "Unitary rotation is not real!"
            circuit += givens_rotation_gate(n_qubits, orb1, orb2, theta)
    return circuit
