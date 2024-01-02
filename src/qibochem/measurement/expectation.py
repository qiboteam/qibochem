import qibo
from qibo import gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z


def expectation(
    circuit: qibo.models.Circuit, hamiltonian: SymbolicHamiltonian, from_samples=False, n_shots=1000
) -> float:
    """
    Calculate expectation value of some Hamiltonian using either the state vector or sample measurements from running a
    quantum circuit

    Args:
        circuit (qibo.models.Circuit): Quantum circuit ansatz
        hamiltonian (SymbolicHamiltonian): Molecular Hamiltonian
        from_samples (Boolean): Whether the expectation value calculation uses samples or the simulated
            state vector. Default: ``False``; Results are from a state vector simulation
        n_shots (int): Number of times the circuit is run if ``from_samples=True``. Default: ``1000``

    Returns:
        Hamiltonian expectation value (float)
    """
    if from_samples:
        from functools import reduce

        total = 0.0
        # Iterate over each term in the Hamiltonian
        for term in hamiltonian.terms:
            # Get the target qubits and basis rotation gates from the Hamiltonian term
            qubits = [int(factor.target_qubit) for factor in term.factors]
            basis = [type(factor.gate) for factor in term.factors]
            # Run a copy of the original circuit to get the output frequencies
            _circuit = circuit.copy()
            _circuit.add(gates.M(*qubits, basis=basis))
            result = _circuit(nshots=n_shots)
            frequencies = result.frequencies(binary=True)
            # Only works for Z terms, raises an error if ham_term has X/Y terms
            # total += SymbolicHamiltonian(
            #              reduce(lambda x, y: x*y, term.factors, 1)
            #          ).expectation_from_samples(frequencies, qubit_map=qubits)
            # Workaround code to handle X/Y terms in the Hamiltonian:
            # Get each Pauli string e.g. X0Y1
            pauli = [factor.name for factor in term.factors]
            # Replace each X and Y symbol with Z; then include the term coefficient
            pauli_z = [Z(int(element[1:])) for element in pauli]
            z_only_ham = SymbolicHamiltonian(term.coefficient * reduce(lambda x, y: x * y, pauli_z, 1.0))
            # Can now apply expectation_from_samples directly
            total += z_only_ham.expectation_from_samples(frequencies, qubit_map=qubits)
        # Add the constant term if present. Note: Energies (in chemistry) are all real values
        total += hamiltonian.constant.real
        return total

    # Expectation value from state vector simulation
    result = circuit(nshots=1)
    state_ket = result.state()
    return hamiltonian.expectation(state_ket)
