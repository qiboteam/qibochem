"""
Test for basis rotation ansatz
"""

import numpy as np
import pytest
from qibo import Circuit, gates

from qibochem.ansatz import basis_rotation
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation

# from qibo.optimizers import optimize


def test_unitary():
    """
    Test for basis_rotation.unitary()
    """
    N = 6
    occ = range(0, 2)
    vir = range(2, 6)

    preset_params = [-0.1, -0.2, -0.3, -0.4]

    U1, theta1 = basis_rotation.unitary(occ, vir)
    U2, theta2 = basis_rotation.unitary(occ, vir, parameters=0.1)
    U3, theta3 = basis_rotation.unitary(occ, vir, parameters=preset_params)

    ref_U2 = np.array(
        [
            [0.99001666, 0.0, 0.099667, 0.0, 0.099667, 0.0],
            [0.0, 0.99001666, 0.0, 0.099667, 0.0, 0.099667],
            [-0.099667, 0.0, 0.99500833, 0.0, -0.00499167, 0.0],
            [0.0, -0.099667, 0.0, 0.99500833, 0.0, -0.00499167],
            [-0.099667, 0.0, -0.00499167, 0.0, 0.99500833, 0.0],
            [0.0, -0.099667, 0.0, -0.00499167, 0.0, 0.99500833],
        ]
    )

    ref_U3 = np.array(
        [
            [0.95041528, 0.0, -0.09834165, 0.0, -0.29502494, 0.0],
            [0.0, 0.9016556, 0.0, -0.19339968, 0.0, -0.38679937],
            [0.09834165, 0.0, 0.99504153, 0.0, -0.01487542, 0.0],
            [0.0, 0.19339968, 0.0, 0.98033112, 0.0, -0.03933776],
            [0.29502494, 0.0, -0.01487542, 0.0, 0.95537375, 0.0],
            [0.0, 0.38679937, 0.0, -0.03933776, 0.0, 0.92132448],
        ]
    )

    identity = np.eye(6)

    assert np.allclose(U1 @ U1.T, identity)
    assert np.allclose(U2 @ U2.T, identity)
    assert np.allclose(U3 @ U3.T, identity)
    assert np.allclose(U1, identity)
    assert np.allclose(U2, ref_U2)
    assert np.allclose(U3, ref_U3)


def test_givens_qr_decompose():
    """
    Test for basis_rotation.givens_qr_decompose()
    """
    N = 6
    occ = range(0, 2)
    vir = range(2, 6)

    U2, theta2 = basis_rotation.unitary(occ, vir, parameters=0.1)
    z_angles, final_U2 = basis_rotation.givens_qr_decompose(U2)

    ref_z = np.array(
        [
            -3.141592653589793,
            -1.5707963267948966,
            -2.356194490192345,
            0.0,
            -1.5707963267948966,
            -1.5207546393123066,
            -1.5707963267948966,
            -1.5707963267948954,
            -3.000171297352484,
            -2.356194490192345,
            0.0,
            -1.5707963267948966,
            -1.5707963267948954,
            -0.09995829685982476,
            -1.5207546393123068,
        ]
    )

    assert np.allclose(z_angles, ref_z)
    assert np.allclose(np.eye(6), final_U2)


def test_basis_rotation_layout():

    N = 10
    ref_A = np.array(
        [
            [0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [1, 0, 6, 0, 15, 0, 28, 0, 45, 0],
            [0, 5, 0, 14, 0, 27, 0, 44, 0, 29],
            [4, 0, 13, 0, 26, 0, 43, 0, 30, 0],
            [0, 12, 0, 25, 0, 42, 0, 31, 0, 16],
            [11, 0, 24, 0, 41, 0, 32, 0, 17, 0],
            [0, 23, 0, 40, 0, 33, 0, 18, 0, 7],
            [22, 0, 39, 0, 34, 0, 19, 0, 8, 0],
            [0, 38, 0, 35, 0, 20, 0, 9, 0, 2],
            [37, -1, 36, -1, 21, -1, 10, -1, 3, -1],
        ]
    )

    A = basis_rotation.basis_rotation_layout(N)
    assert np.allclose(A, ref_A)
