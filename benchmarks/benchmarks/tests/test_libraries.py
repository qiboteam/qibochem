"""Check that execution of circuits from external simulation libraries agrees with Qibo."""

import itertools

import numpy as np
import pytest
from qibo import gates, models

from benchmarks import libraries
from benchmarks.circuits import qasm, qibo


def assert_circuit_execution(backend, qasm_circuit, qibo_circuit_iter, atol=None):
    if atol is None:
        if backend.get_precision() == "single":
            atol = 1e-5
        else:
            atol = 1e-10

    # add random RX gates before circuit so that initial state is not trivial
    nqubits = qasm_circuit.nqubits
    theta = np.random.random(nqubits)
    qasm_code = qasm_circuit.to_qasm(theta=theta)

    # execute circuit using backend
    circuit = backend.from_qasm(qasm_code)
    final_state = backend(circuit)
    final_state = backend.transpose_state(final_state)

    # execute circuit using qibo
    assert qibo_circuit_iter.nqubits == nqubits
    target_circuit = models.Circuit(nqubits)
    target_circuit.add(gates.RX(i, theta=t) for i, t in enumerate(theta))
    target_circuit.add(qibo_circuit_iter)
    target_state = target_circuit()

    # check fidelity instead of absolute states due to different definitions
    # of the phase of U gates in different backends
    fidelity = np.abs(np.conj(target_state).dot(np.array(final_state)))
    np.testing.assert_allclose(fidelity, 1.0, atol=atol)


@pytest.mark.parametrize("nlayers", ["1", "4"])
@pytest.mark.parametrize("gate, qibo_gate", [("h", "H"), ("x", "X"), ("y", "Y"), ("z", "Z")])
def test_one_qubit_gate(nqubits, library, nlayers, gate, qibo_gate):
    qasm_circuit = qasm.OneQubitGate(nqubits, nlayers=nlayers, gate=gate)
    target_circuit = qibo.OneQubitGate(nqubits, nlayers=nlayers, gate=qibo_gate)
    backend = libraries.get(library)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize(
    "gate,qibo_gate,params",
    [
        ("rx", "RX", {"theta": 0.1}),
        ("ry", "RY", {"theta": 0.3}),
        ("rz", "RZ", {"theta": 0.2}),
        ("u1", "U1", {"theta": 0.3}),
        ("u2", "U2", {"phi": 0.2, "lam": 0.3}),
        ("u3", "U3", {"theta": 0.1, "phi": 0.2, "lam": 0.3}),
    ],
)
def test_one_qubit_gate_parametrized(nqubits, library, gate, qibo_gate, params):
    if gate in {"u1", "u2", "u3"} and library == "tfq":
        pytest.skip("Skipping {} test because it is not supported by {}." "".format(gate, library))
    order = ["theta", "phi", "lam"]
    angles = ",".join(str(params.get(n)) for n in order if n in params)
    qasm_circuit = qasm.OneQubitGate(nqubits, gate=gate, angles=angles)
    target_circuit = qibo.OneQubitGate(nqubits, gate=qibo_gate, **params)
    backend = libraries.get(library)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize("nlayers", ["1", "4"])
@pytest.mark.parametrize("gate,qibo_gate", [("cx", "CNOT"), ("swap", "SWAP"), ("cz", "CZ")])
def test_two_qubit_gate(nqubits, library, nlayers, gate, qibo_gate):
    qasm_circuit = qasm.TwoQubitGate(nqubits, nlayers=nlayers, gate=gate)
    target_circuit = qibo.TwoQubitGate(nqubits, nlayers=nlayers, gate=qibo_gate)
    backend = libraries.get(library)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize(
    "gate,qibo_gate,params",
    [
        ("cu1", "CU1", {"theta": 0.3}),
        # ("cu2", "CU2", {"phi": 0.1, "lam": 0.3}), # not supported by OpenQASM
        ("cu3", "CU3", {"theta": 0.1, "phi": 0.2, "lam": 0.3}),
    ],
)
def test_two_qubit_gate_parametrized(nqubits, library, gate, qibo_gate, params):
    if gate in {"cu1", "cu2", "cu3"} and library == "tfq":
        pytest.skip("Skipping {} test because it is not supported by {}." "".format(gate, library))
    if gate in {"cu3"} and library == "projectq":
        pytest.skip("Skipping {} test because it is not supported by {}." "".format(gate, library))

    order = ["theta", "phi", "lam"]
    angles = ",".join(str(params.get(n)) for n in order if n in params)
    qasm_circuit = qasm.TwoQubitGate(nqubits, gate=gate, angles=angles)
    target_circuit = qibo.TwoQubitGate(nqubits, gate=qibo_gate, **params)
    backend = libraries.get(library)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize("swaps", ["False", "True"])
def test_qft(nqubits, library, swaps, library_options):
    qasm_circuit = qasm.QFT(nqubits, swaps=swaps)
    target_circuit = qibo.QFT(nqubits, swaps=swaps)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize("nlayers", ["2", "5"])
def test_variational(nqubits, library, nlayers, library_options):
    qasm_circuit = qasm.VariationalCircuit(nqubits, nlayers=nlayers)
    target_circuit = qibo.VariationalCircuit(nqubits, nlayers=nlayers)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


def test_bernstein_vazirani(nqubits, library, library_options):
    qasm_circuit = qasm.BernsteinVazirani(nqubits)
    target_circuit = qibo.BernsteinVazirani(nqubits)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


def test_hidden_shift(nqubits, library, library_options):
    shift = "".join(str(x) for x in np.random.randint(0, 2, size=(nqubits,)))
    qasm_circuit = qasm.HiddenShift(nqubits, shift=shift)
    target_circuit = qibo.HiddenShift(nqubits, shift=shift)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


def test_qaoa_circuit(library, library_options):
    if library in {"qibo", "qcgpu"}:
        pytest.skip(f"{library} does not have built-in RZZ gate.")
    import pathlib

    folder = str(pathlib.Path(__file__).with_name("graphs") / "testgraph8.json")
    qasm_circuit = qasm.QAOA(8, graph=folder)
    target_circuit = qibo.QAOA(8, graph=folder)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize("depth", ["2", "5", "10"])
def test_supremacy_circuit(nqubits, library, depth, library_options):
    qasm_circuit = qasm.SupremacyCircuit(nqubits, depth=depth)
    target_circuit = qibo.SupremacyCircuit(nqubits, depth=depth)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize("simtime", ["1", "2.5"])
def test_basis_change(nqubits, library, simtime, library_options):
    qasm_circuit = qasm.BasisChange(nqubits, simulation_time=simtime)
    target_circuit = qibo.BasisChange(nqubits, simulation_time=simtime)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)


@pytest.mark.parametrize("depth", ["2", "5", "8"])
def test_quantum_volume(nqubits, library, depth, library_options):
    if library == "tfq":
        pytest.skip("Skipping qv test because it is not supported by {}." "".format(library))
    qasm_circuit = qasm.QuantumVolume(nqubits, depth=depth)
    target_circuit = qibo.QuantumVolume(nqubits, depth=depth)
    backend = libraries.get(library, library_options)
    assert_circuit_execution(backend, qasm_circuit, target_circuit)
