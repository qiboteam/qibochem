"""
Test Expectation class functions
"""

import numpy as np
from qibo import gates, Circuit
from openfermion.ops import QubitOperator

from qibochem.measurement import basis_rotate, expectation

# basis_rotate.py unit tests
def test_measure_rotate_basis():
    """Basis rotations"""
    circuit_1 = Circuit(3)
    circuit_1.add(gates.H(0))
    circuit_1.add(gates.M(0))
    circuit_1.add(gates.S(1).dagger())
    circuit_1.add(gates.H(1))
    circuit_1.add(gates.M(1))
    circuit_1.add(gates.M(2))

    pauli_op = QubitOperator('X0 Y1 Z2')
    basis_rotate_shots = basis_rotate.measure_rotate_basis(list(pauli_op.terms)[0], 3)

    assert circuit_1.summary() == basis_rotate_shots.summary()

# expectation.py unit tests
def test_pauli_expectation_shots():
    """Pauli expectations"""
    # Unit test of pauli expectation
    # Test to check function, first assertion checks subtraction of count to bitcount, cloned circuit checks addition of count to bitcount
    circuit_1 = Circuit(2)
    circuit_1.add(gates.X(0))
    circuit_1.add(gates.M(0))

    expectation_pauli = expectation.pauli_expectation_shots(circuit_1, 1000)
    assert expectation_pauli == -1

    circuit_2 = circuit_1.copy()
    circuit_2.add(gates.X(1))
    circuit_2.add(gates.M(1))

    expectation_pauli = expectation.pauli_expectation_shots(circuit_2, 1000)
    assert expectation_pauli == 1

def test_circuit_expectation_shots():
    """Circuit expectations"""
    # Tests all the basis by feeding a simple circuit and checking the results

    # Unit test of no basis
    circuit_no = Circuit(1)
    circuit_no.add(gates.X(0))
    of_qubham = QubitOperator('')

    expectation_no = expectation.circuit_expectation_shots(circuit_no, of_qubham, 1000)
    assert expectation_no == 1

    # Unit test of X basis
    circuit_x = Circuit(1)
    circuit_x.add(gates.H(0))
    circuit_x.add(gates.Z(0))
    of_qubham = QubitOperator('X0')

    expectation_x = expectation.circuit_expectation_shots(circuit_x, of_qubham, 1000)
    assert expectation_x == -1

    # Unit test of y basis
    circuit_y = Circuit(1)
    circuit_y.add(gates.H(0))
    circuit_y.add(gates.S(0))
    circuit_y.add(gates.X(0))
    of_qubham = QubitOperator('Y0')

    expectation_y = expectation.circuit_expectation_shots(circuit_y, of_qubham, 1000)
    assert expectation_y == -1

    # Unit test of Z basis
    circuit_z = Circuit(1)
    circuit_z.add(gates.X(0))
    of_qubham = QubitOperator('Z0')

    expectation_z = expectation.circuit_expectation_shots(circuit_z, of_qubham, 1000)
    assert expectation_z == -1

# Executes the test when file is called using python
if __name__ == '__main__':
    test_measure_rotate_basis()
    test_pauli_expectation_shots()
    test_circuit_expectation_shots()