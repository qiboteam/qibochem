import numpy as np
import pytest
from qibo import gates
from qibo.noise import DepolarizingError, NoiseModel
from qibo.optimizers import optimize

from qibochem.ansatz import he_circuit
from qibochem.driver import Molecule
from qibochem.measurement import expectation


def test_he_circuit():
    n_qubits = 4
    n_layers = 1
    rotation_gates = ["RX"]
    entanglement_gate = "CNOT"
    gate_list = []
    for _ in range(n_layers):
        # Rotation gates
        gate_list += [
            getattr(gates, rotation_gate)(_i, 0.0) for rotation_gate in rotation_gates for _i in range(n_qubits)
        ]
        # Entanglement gates
        gate_list += [getattr(gates, entanglement_gate)(_i, _i + 1) for _i in range(n_qubits - 1)]
    # Test function
    test_circuit = he_circuit(n_qubits, n_layers, rotation_gates, entanglement_gate)

    # Check gates are correct
    assert all(
        control.name == test.name and control.target_qubits == test.target_qubits
        for control, test in zip(gate_list, list(test_circuit.queue))
    )
    # Check that only two parametrised gates
    assert len(test_circuit.get_parameters()) == n_layers * n_qubits * len(rotation_gates)


def test_he_circuit_noise_model():
    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    n_qubits = 4
    n_layers = 1
    rotation_gates = ["RX"]
    entanglement_gate = "CNOT"
    # Test function
    test_circuit = he_circuit(n_qubits, n_layers, rotation_gates, entanglement_gate, noise_model=noise_model)

    for _i in range(test_circuit.nqubits):
        test_circuit.add(gates.M(_i))

    counts = test_circuit(nshots=int(1e3)).frequencies()
    probs = {}
    probs = {}
    for bitstring, count in counts.items():
        probs[bitstring] = count / sum(counts.values())

    ground_truth = {format(i, f"0{n_qubits}b"): 1 / 2**n_qubits for i in range(2**n_qubits)}
    # assert keys match
    assert probs.keys() == ground_truth.keys()
    # assert values
    for key in probs:
        assert np.allclose(probs[key], ground_truth[key], atol=1e-1)


def test_vqe_he_ansatz():
    # Loss function for VQE
    def electronic_energy(parameters, circuit, hamiltonian):
        circuit.set_parameters(parameters)
        return expectation(circuit, hamiltonian)

    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    mol.run_pyscf()
    hamiltonian = mol.hamiltonian()

    n_layers = 2
    n_qubits = mol.nso
    rotation_gates = None
    entanglement_gate = "CNOT"
    circuit = he_circuit(n_qubits, n_layers, rotation_gates, entanglement_gate)

    n_parameters = len(circuit.get_parameters())
    thetas = np.full(n_parameters, 0.25 * np.pi)
    best, params, extra = optimize(electronic_energy, thetas, args=(circuit, hamiltonian), method="BFGS")

    exact_result = -1.136189454
    assert best == pytest.approx(exact_result), f"{best} != exact_result (= {exact_result})"
