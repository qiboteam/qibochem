"""
Test Expectation class functions
"""

import numpy as np
from qibo import gates, Circuit
from openfermion.ops import QubitOperator

from qibochem.ansatz import ucc
from qibochem.ansatz import hardware_efficient

# hardware_efficient unit test
def test_hea_su2():
    """hea_su2"""
    # Tests function by comparing it with a hardcoded answer |0..0> states return itself on hea_su2
    # Can be further expanded by testing other initial and final states
    for i in range(2,6):
        test_circuit = Circuit(i)
        gate_list_hea_su2 = hardware_efficient.hea_su2(i,i)
        for j in range (len(gate_list_hea_su2)):
            test_circuit.add(gate_list_hea_su2[j])
        result = test_circuit(nshots=1)

        test_result = np.zeros(2**i, dtype=complex)
        test_result[0] += 1

        assert np.all(result.state() == test_result)

# ucc unit test
def test_expi_pauli():
    test_circuit, test_coeff = ucc.expi_pauli(n_qubits=3, theta=0, pauli_string=QubitOperator('X0 Y1 Z2'))


# Executes the test when file is called using python
if __name__ == '__main__':
    test_hea_su2()
    test_expi_pauli()
