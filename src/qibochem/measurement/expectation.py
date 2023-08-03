# from qibo import gates
# from qibo.models import Circuit

import qibo
from qibo.hamiltonians import SymbolicHamiltonian

from qibochem.measurement.basis_rotate import measure_rotate_basis


def pauli_expectation_shots(qc, nshots):
    """
    Calculates expectation value of circuit for pauli string from shots

    Args:
        qc: quantum circuit with appropriate rotations and measure gates
        nshots: number of shots

    Returns:
        expectation: expectation value
    """
    result = qc.execute(nshots=nshots)
    meas = result.frequencies(binary=True)

    bitcount = 0
    for bitstring, count in meas.items():
        if bitstring.count("1") % 2 == 0:
            bitcount += count
        else:
            bitcount -= count
    expectation = bitcount / nshots
    return expectation


def circuit_expectation_shots(qc, of_qubham, nshots=1000):
    """
    Calculates expectation value of qubit hamiltonian w.r.t. circuit ansatz using shots

    Args:
        qc: Quantum circuit with ansatz
        of_qubham: Molecular Hamiltonian given as an OpenFermion QubitOperator

    Returns:
        Expectation value of of_qubham w.r.t. circuit ansatz
    """
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
        if len(pauli_op[0]) == 0:  # no pauli obs to measure,
            expectation += coeffs[istring]
            # print(coeffs[istring])
        else:
            # add rotation gates to rotate pauli observable to computational basis
            # i.e. X to Z, Y to Z
            meas_qc = measure_rotate_basis(pauli_op[0], nqubits)
            full_qc = qc + meas_qc
            # print(full_qc.draw())
            pauli_op_exp = pauli_expectation_shots(full_qc, nshots)
            expectation += coeffs[istring] * pauli_op_exp
            ## print(pauli_op, pauli_op_exp, coeffs[istring])
        istring += 1

    return expectation


def expectation(
    circuit: qibo.models.Circuit, hamiltonian: SymbolicHamiltonian, from_samples=False, n_shots=1000
) -> float:
    """
    Calculate expectation value of Hamiltonian using either the state vector from running a
        circuit, or the frequencies of the resultant binary string results

    Args:
        circuit (qibo.models.Circuit): Quantum circuit ansatz
        hamiltonian (SymbolicHamiltonian): Molecular Hamiltonian
        from_samples: Whether the expectation value calculation uses samples or the simulated
            state vector. Default: False, state vector simulation
        n_shots: Number of times the circuit is run for the from_samples=True case

    Returns:
        Hamiltonian expectation value (float)
    """
    if from_samples:
        raise NotImplementedError("expectation function only works with state vector")
    # TODO: Rough code for expectation_from_samples if issue resolved
    # Yet to test!!!!!
    #
    # from functools import reduce
    # total = 0.0
    # Iterate over each term in the Hamiltonian
    # for term in hamiltonian.terms:
    #     # Get the basis rotation gates and target qubits from the Hamiltonian term
    #     qubits = [factor.target_qubit for factor in term.factors]
    #     basis = [type(factor.gate) for factor in term.factors]
    #     # Run a copy of the initial circuit to get the output frequencies
    #     _circuit = circuit.copy()
    #     _circuit.add(gates.M(*qubits, basis=basis))
    #     result = _circuit(nshots=n_shots)
    #     frequencies = result.frequencies(binary=True)
    #     # Only works for Z terms, raises an error if ham_term has X/Y terms
    #     total += SymbolicHamiltonian(
    #                  reduce(lambda x, y: x*y, term.factors, 1)
    #              ).expectation_from_samples(frequencies, qubit_map=qubits)
    # return total

    # Expectation value from state vector simulation
    result = circuit(nshots=1)
    state_ket = result.state()
    return hamiltonian.expectation(state_ket)
