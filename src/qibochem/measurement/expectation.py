
# from qibo import gates
# from qibo.models import Circuit

from qibochem.measurement.basis_rotate import measure_rotate_basis

def pauli_expectation_shots(qc, nshots):
    '''
    Calculates expectation value of circuit for pauli string from shots

    Args:
        qc: quantum circuit with appropriate rotations and measure gates
        nshots: number of shots

    Returns:
        expectation: expectation value
    '''
    result = qc.execute(nshots=nshots)
    meas = result.frequencies(binary=True)

    bitcount = 0
    for bitstring, count in meas.items():
        if bitstring.count('1') % 2 == 0:
            bitcount += count
        else:
            bitcount -= count
    expectation = bitcount/nshots
    return expectation

def circuit_expectation_shots(qc, of_qubham, nshots=1000):
    '''
    Calculates expectation value of qubit hamiltonian w.r.t. circuit ansatz using shots

    Args:
        qc: Quantum circuit with ansatz
        of_qubham: Molecular Hamiltonian given as an OpenFermion QubitOperator 

    Returns:
        Expectation value of of_qubham w.r.t. circuit ansatz
    '''
    # nterms = len(of_qubham.terms)
    nqubits = qc.nqubits
    # print(nterms)
    coeffs = []
    strings = []

    expectation = 0
    for qubop in of_qubham.terms:
        coeffs.append(of_qubham.terms[qubop])
        strings.append([qubop])

    istring = 0
    for pauli_op in strings:

        if len(pauli_op[0]) == 0: # no pauli obs to measure,
            expectation += coeffs[istring]
            # print(coeffs[istring])
        else:
            # add rotation gates to rotate pauli observable to computational basis
            # i.e. X to Z, Y to Z
            meas_qc = measure_rotate_basis(pauli_op[0], nqubits)
            full_qc = qc + meas_qc
            # print(full_qc.draw())
            pauli_op_exp = pauli_expectation_shots(full_qc, nshots)
            expectation += coeffs[istring]*pauli_op_exp
            ## print(pauli_op, pauli_op_exp, coeffs[istring])
        istring += 1

    return expectation
