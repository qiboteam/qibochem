"""Benchmark scripts."""

import time

from qibo.backends import GlobalBackend

from benchmarks.logger import JsonLogger

# Local script for Qibochem benchmarking
from qibochem.ansatz import ucc_ansatz
from qibochem.driver import Molecule


# Original 'nqubits' argument replaced with 'n_hydrogens'
# def circuit_benchmark(nqubits, backend, circuit_name, circuit_options=None,
def circuit_benchmark(
    n_hydrogens,
    backend,  # circuit_name, circuit_options=None,
    nreps=1,
    nshots=None,
    transfer=False,
    precision="double",
    memory=None,
    threading=None,
    filename=None,
    platform=None,
):
    """Runs benchmark for different circuit types.

    See ``benchmarks/main.py`` for documentation of each argument.
    """
    if backend == "qibojit" and threading is not None:
        from benchmarks.utils import select_numba_threading

        threading = select_numba_threading(threading)

    if backend in {"qibotf", "tensorflow"} and memory is not None:
        from benchmarks.utils import limit_gpu_memory

        memory = limit_gpu_memory(memory)

    logs = JsonLogger(filename)
    # logs.log(nqubits=nqubits, nreps=nreps, nshots=nshots, transfer=transfer,
    logs.log(
        nqubits=2 * n_hydrogens,
        nreps=nreps,
        nshots=nshots,
        transfer=transfer,
        numba_threading=threading,
        gpu_memory=memory,
    )

    start_time = time.time()
    import qibo

    logs.log(import_time=time.time() - start_time)

    qibo.set_backend(backend=backend, platform=platform)
    qibo.set_precision(precision)
    logs.log(
        backend=qibo.get_backend(),
        platform=GlobalBackend().platform,
        precision=qibo.get_precision(),
        device=qibo.get_device(),
        version=qibo.__version__,
    )

    # Construct UCC circuit ansatz using some hydrogen chain
    # 1. Get molecule first
    start_time = time.time()
    # mol = build_molecule(n_hydrogens)
    # assert n_hydrogens % 2 == 0, f"{n_hydrogens} must be even"
    hh_bond_length = 0.75  # angstroms
    geom = [("H", (0.0, 0.0, _i * hh_bond_length)) for _i in range(n_hydrogens)]
    mol = Molecule(geom)
    mol.run_pyscf()
    logs.log(pyscf_time=time.time() - start_time)
    # Build the actual UCC ansatz
    start_time = time.time()
    circuit = ucc_ansatz(mol)
    logs.log(creation_time=time.time() - start_time)

    # from benchmarks import circuits
    # gates = circuits.get(circuit_name, nqubits, circuit_options, qibo=True)
    # logs.log(circuit=circuit_name, circuit_options=str(gates))
    # start_time = time.time()
    # circuit = qibo.models.Circuit(nqubits)
    # circuit.add(gates)
    # if nshots is not None:
    #     # add measurement gates
    #     circuit.add(qibo.gates.M(*range(nqubits)))
    # logs.log(creation_time=time.time() - start_time)

    start_time = time.time()
    result = circuit(nshots=nshots)
    logs.log(dry_run_time=time.time() - start_time)
    start_time = time.time()
    if transfer:
        result = result.numpy()
    logs.log(dry_run_transfer_time=time.time() - start_time)
    dtype = str(result.state().dtype)
    del result
    del circuit

    creation_times, simulation_times, transfer_times = [], [], []
    for _ in range(nreps):
        start_time = time.time()
        circuit = ucc_ansatz(mol)

        # circuit = qibo.models.Circuit(nqubits)
        # circuit.add(gates)
        # if nshots is not None:
        #     # add measurement gates
        #     circuit.add(qibo.gates.M(*range(nqubits)))

        creation_times.append(time.time() - start_time)
        start_time = time.time()
        result = circuit(nshots=nshots)
        simulation_times.append(time.time() - start_time)
        start_time = time.time()
        if transfer:
            result = result.numpy()
        transfer_times.append(time.time() - start_time)
        del result
        del circuit

    logs.log(
        dtype=dtype, creation_times=creation_times, simulation_times=simulation_times, transfer_times=transfer_times
    )
    logs.average("creation_times")
    logs.average("simulation_times")
    logs.average("transfer_times")

    if nshots is not None:
        result = circuit(nshots=nshots)
        start_time = time.time()
        freqs = result.frequencies()
        logs.log(measurement_time=time.time() - start_time)
        del result
    else:
        logs.log(measurement_time=0)
        logs.dump()

    return logs


def library_benchmark(
    nqubits, library, circuit_name, circuit_options=None, library_options=None, precision=None, nreps=1, filename=None
):
    """Runs benchmark for different quantum simulation libraries.

    See ``benchmarks/compare.py`` for documentation of each argument.
    """
    logs = JsonLogger(filename)
    logs.log(nqubits=nqubits, nreps=nreps)

    start_time = time.time()
    from benchmarks import libraries

    backend = libraries.get(library, library_options)
    logs.log(import_time=time.time() - start_time)
    logs.log(library_options=library_options)
    if precision is not None:
        backend.set_precision(precision)

    logs.log(
        library=backend.name,
        precision=backend.get_precision(),
        device=backend.get_device(),
        version=backend.__version__,
    )

    from benchmarks import circuits

    gates = circuits.get(circuit_name, nqubits, circuit_options)
    logs.log(circuit=circuit_name, circuit_options=str(gates))
    start_time = time.time()
    circuit = backend.from_qasm(gates.to_qasm())
    logs.log(creation_time=time.time() - start_time)

    start_time = time.time()
    result = backend(circuit)
    logs.log(dry_run_time=time.time() - start_time)
    dtype = str(result.dtype)
    del result

    simulation_times = []
    for _ in range(nreps):
        start_time = time.time()
        result = backend(circuit)
        simulation_times.append(time.time() - start_time)
        del result

    logs.log(dtype=dtype, simulation_times=simulation_times)
    logs.average("simulation_times")
    logs.dump()
    return logs


def evolution_benchmark(
    nqubits, dt, solver, backend, platform=None, nreps=1, precision="double", dense=False, filename=None
):
    """Performs adiabatic evolution with critical TFIM as the hard Hamiltonian."""
    logs = JsonLogger(filename)
    logs.log(nqubits=nqubits, nreps=nreps, dt=dt, solver=solver, dense=dense)

    start_time = time.time()
    import qibo

    logs.log(import_time=time.time() - start_time)

    qibo.set_backend(backend=backend, platform=platform)
    qibo.set_precision(precision)
    logs.log(
        backend=qibo.get_backend(),
        platform=GlobalBackend().platform,
        precision=qibo.get_precision(),
        device=qibo.get_device(),
        threads=qibo.get_threads(),
        version=qibo.__version__,
    )

    from qibo import hamiltonians, models

    start_time = time.time()
    h0 = hamiltonians.X(nqubits, dense=dense)
    h1 = hamiltonians.TFIM(nqubits, h=1.0, dense=dense)
    logs.log(hamiltonian_creation_time=time.time() - start_time)

    start_time = time.time()
    evolution = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt, solver=solver)
    logs.log(evolution_creation_time=time.time() - start_time)

    start_time = time.time()
    result = evolution(final_time=1.0)
    logs.log(dry_run_time=time.time() - start_time)
    dtype = str(result.dtype)
    del result

    simulation_times = []
    for _ in range(nreps):
        start_time = time.time()
        result = evolution(final_time=1.0)
        simulation_times.append(time.time() - start_time)
    logs.log(dtype=dtype, simulation_times=simulation_times)
    logs.average("simulation_times")
    logs.dump()
    return logs
