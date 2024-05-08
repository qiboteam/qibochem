import pytest

from benchmarks.scripts import circuit_benchmark, evolution_benchmark


def assert_logs(logs, nqubits, backend, nreps=1):
    assert logs[-1]["nqubits"] == nqubits
    assert logs[-1]["backend"] == backend
    assert logs[-1]["simulation_times_mean"] >= 0
    assert logs[-1]["transfer_times_mean"] >= 0
    assert len(logs[-1]["simulation_times"]) == nreps
    assert len(logs[-1]["transfer_times"]) == nreps


@pytest.mark.parametrize("nreps", [1, 5])
@pytest.mark.parametrize("nlayers", ["1", "4"])
@pytest.mark.parametrize("gate", ["H", "X", "Y", "Z"])
def test_one_qubit_gate_benchmark(nqubits, backend, transfer, nreps, nlayers, gate):
    logs = circuit_benchmark(
        nqubits,
        backend,
        circuit_name="one-qubit-gate",
        nreps=nreps,
        transfer=transfer,
        circuit_options=f"gate={gate},nlayers={nlayers}",
    )
    assert_logs(logs, nqubits, backend, nreps)
    target_options = f"nqubits={nqubits}, nlayers={nlayers}, "
    target_options += f"gate={gate}, params={{}}"
    assert logs[-1]["circuit"] == "one-qubit-gate"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize(
    "gate,params",
    [
        ("RX", "theta=0.1"),
        ("RZ", "theta=0.2"),
        ("U1", "theta=0.3"),
        ("U2", "phi=0.2,lam=0.3"),
        ("U3", "theta=0.1,phi=0.2,lam=0.3"),
    ],
)
def test_one_qubit_gate_param_benchmark(nqubits, backend, gate, params):
    logs = circuit_benchmark(nqubits, backend, circuit_name="one-qubit-gate", circuit_options=f"gate={gate},{params}")
    assert_logs(logs, nqubits, backend)
    target_options = f"nqubits={nqubits}, nlayers=1, gate={gate}"
    paramdict = {}
    for param in params.split(","):
        k, v = param.split("=")
        paramdict[k] = v
    target_options = f"{target_options}, params={paramdict}"
    assert logs[-1]["circuit"] == "one-qubit-gate"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize("nreps", [1, 5])
@pytest.mark.parametrize("nlayers", ["1", "4"])
@pytest.mark.parametrize("gate", ["CNOT", "SWAP", "CZ"])
def test_two_qubit_gate_benchmark(nqubits, backend, transfer, nreps, nlayers, gate):
    logs = circuit_benchmark(
        nqubits,
        backend,
        circuit_name="two-qubit-gate",
        nreps=nreps,
        transfer=transfer,
        circuit_options=f"gate={gate},nlayers={nlayers}",
    )
    assert_logs(logs, nqubits, backend, nreps)
    target_options = f"nqubits={nqubits}, nlayers={nlayers}, "
    target_options += f"gate={gate}, params={{}}"
    assert logs[-1]["circuit"] == "two-qubit-gate"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize(
    "gate,params",
    [
        ("CRX", "theta=0.1"),
        ("CRZ", "theta=0.2"),
        ("CU1", "theta=0.3"),
        ("CU2", "phi=0.2,lam=0.3"),
        ("CU3", "theta=0.1,phi=0.2,lam=0.3"),
        ("fSim", "theta=0.1,phi=0.2"),
    ],
)
def test_two_qubit_gate_param_benchmark(nqubits, backend, gate, params):
    logs = circuit_benchmark(nqubits, backend, circuit_name="two-qubit-gate", circuit_options=f"gate={gate},{params}")
    assert_logs(logs, nqubits, backend)
    target_options = f"nqubits={nqubits}, nlayers=1, gate={gate}"
    paramdict = {}
    for param in params.split(","):
        k, v = param.split("=")
        paramdict[k] = v
    target_options = f"{target_options}, params={paramdict}"
    assert logs[-1]["circuit"] == "two-qubit-gate"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize("nreps", [1, 5])
@pytest.mark.parametrize("swaps", [False, True])
def test_qft_benchmark(nqubits, backend, transfer, nreps, swaps):
    logs = circuit_benchmark(
        nqubits, backend, circuit_name="qft", nreps=nreps, transfer=transfer, circuit_options=f"swaps={swaps}"
    )
    assert_logs(logs, nqubits, backend, nreps)
    target_options = f"nqubits={nqubits}, swaps={swaps}"
    assert logs[-1]["circuit"] == "qft"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize("varlayer", [False, True])
def test_variational_benchmark(nqubits, backend, varlayer):
    logs = circuit_benchmark(nqubits, backend, circuit_name="variational", circuit_options=f"varlayer={varlayer}")
    assert_logs(logs, nqubits, backend)
    target_options = f"nqubits={nqubits}, nlayers=1, seed=123, varlayer={varlayer}"
    assert logs[-1]["circuit"] == "variational"
    assert logs[-1]["circuit_options"] == target_options


def test_bernstein_vazirani_benchmark(nqubits, backend):
    logs = circuit_benchmark(nqubits, backend, circuit_name="bv")
    assert_logs(logs, nqubits, backend)
    assert logs[-1]["circuit"] == "bv"
    assert logs[-1]["circuit_options"] == f"nqubits={nqubits}"


@pytest.mark.parametrize("random", [True, False])
def test_hidden_shift_benchmark(nqubits, backend, random):
    shift = "" if random else nqubits * "0"
    logs = circuit_benchmark(nqubits, backend, circuit_name="hs", circuit_options=f"shift={shift}")
    assert_logs(logs, nqubits, backend)
    target_options = f"nqubits={nqubits}, shift={shift}"
    assert logs[-1]["circuit"] == "hs"
    assert logs[-1]["circuit_options"] == target_options


def test_qaoa_benchmark(backend):
    logs = circuit_benchmark(4, backend, circuit_name="qaoa")
    assert_logs(logs, 4, backend)
    target_options = f"nqubits=4, nparams=2, graph=, seed=123"
    assert logs[-1]["circuit"] == "qaoa"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize("depth", ["2", "5", "10"])
def test_supremacy_benchmark(nqubits, backend, depth):
    logs = circuit_benchmark(nqubits, backend, circuit_name="supremacy", circuit_options=f"depth={depth}")
    assert_logs(logs, nqubits, backend)
    target_options = f"nqubits={nqubits}, depth={depth}, seed=123"
    assert logs[-1]["circuit"] == "supremacy"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize("simtime", ["1", "2.5"])
def test_basis_change_benchmark(nqubits, backend, simtime):
    logs = circuit_benchmark(nqubits, backend, circuit_name="bc", circuit_options=f"simulation_time={simtime}")
    assert_logs(logs, nqubits, backend)
    target_options = f"nqubits={nqubits}, simulation_time={simtime}, seed=123"
    assert logs[-1]["circuit"] == "bc"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize("depth", ["2", "5", "8"])
def test_quantum_volume_benchmark(nqubits, backend, depth):
    logs = circuit_benchmark(nqubits, backend, circuit_name="qv", circuit_options=f"depth={depth}")
    assert_logs(logs, nqubits, backend)
    target_options = f"nqubits={nqubits}, depth={depth}, seed=123"
    assert logs[-1]["circuit"] == "qv"
    assert logs[-1]["circuit_options"] == target_options


@pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("dense", [False, True])
def test_adiabatic_evolution_benchmark(nqubits, dt, backend, dense, solver="exp"):
    logs = evolution_benchmark(nqubits, dt, solver, backend, dense=dense)
    assert logs[-1]["nqubits"] == nqubits
    assert logs[-1]["dt"] == dt
    assert logs[-1]["backend"] == backend
    assert logs[-1]["dense"] == dense
