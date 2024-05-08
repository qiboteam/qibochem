NQUBITS = "3,4,5"
MAX_QUBITS = "0,1,2,3,4"
QIBO_BACKENDS = "qibojit,tensorflow,numpy"
LIBRARIES = "qibo,qiskit,cirq,qsim,tfq,qulacs,projectq,hybridq"


def pytest_addoption(parser):
    parser.addoption("--nqubits", type=str, default=NQUBITS)
    parser.addoption("--max-qubits", type=str, default=MAX_QUBITS)
    parser.addoption("--qibo-backends", type=str, default=QIBO_BACKENDS)
    parser.addoption("--libraries", type=str, default=LIBRARIES)
    parser.addoption("--add", type=str, default="")


def pytest_generate_tests(metafunc):
    nqubits = [int(n) for n in metafunc.config.option.nqubits.split(",")]
    library_options = [f"max_qubits={n}" for n in metafunc.config.option.max_qubits.split(",")]
    backends = metafunc.config.option.qibo_backends.split(",")
    libraries = metafunc.config.option.libraries.split(",")
    additional = metafunc.config.option.add
    if additional:
        libraries.extend(additional.split(","))

    if "nqubits" in metafunc.fixturenames:
        metafunc.parametrize("nqubits", nqubits)
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", backends)
    if "library" in metafunc.fixturenames:
        metafunc.parametrize("library", libraries)
    if "library_options" in metafunc.fixturenames:
        metafunc.parametrize("library_options", library_options)
    if "transfer" in metafunc.fixturenames:
        metafunc.parametrize("transfer", [False, True])
