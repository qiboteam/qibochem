"""Tests for ground-state QSCI."""

import numpy as np
import pytest
from openfermion import QubitOperator, get_sparse_operator
from qibo import models, gates

from qibochem.selected_ci.qsci import QSCI, QSCIConfig, _project_hamiltonian_subspace


def _test_hamiltonian() -> QubitOperator:
    return (
        -0.3 * QubitOperator(())
        + 0.7 * QubitOperator("Z0")
        - 0.4 * QubitOperator("Z1")
        + 0.5 * QubitOperator("X0 X1")
        + 0.2 * QubitOperator("Y0 Y1")
    )


def _exact_ground_energy(qubit_ham: QubitOperator, n_qubits: int) -> float:
    matrix = get_sparse_operator(qubit_ham, n_qubits=n_qubits).toarray()
    return float(np.linalg.eigvalsh(matrix)[0])


def _exact_energies(qubit_ham: QubitOperator, n_qubits: int) -> np.ndarray:
    matrix = get_sparse_operator(qubit_ham, n_qubits=n_qubits).toarray()
    return np.linalg.eigvalsh(matrix)


def test_top_r_selection_deterministic_tiebreak():
    hamiltonian = _test_hamiltonian()
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=1))

    result = runner.run_from_samples(["10", "01"])
    assert result.selected_bitstrings == ["01"]


def test_particle_number_postselection_filters_bitstrings():
    hamiltonian = _test_hamiltonian()
    runner = QSCI(
        hamiltonian,
        n_qubits=2,
        config=QSCIConfig(r=4, postselect=True, n_electrons=1),
    )

    result = runner.run_from_samples(["00", "01", "10", "11", "11", "10"])

    assert set(result.counts.keys()) == {"01", "10"}
    assert all(bitstring.count("1") == 1 for bitstring in result.selected_bitstrings)


def test_projected_matrix_is_hermitian():
    hamiltonian = _test_hamiltonian()
    basis = ["00", "11", "01"]

    projected = _project_hamiltonian_subspace(hamiltonian, basis)
    assert np.allclose(projected, projected.conj().T)


def test_qsci_variational_upper_bound_small_system():
    hamiltonian = _test_hamiltonian()
    exact = _exact_ground_energy(hamiltonian, n_qubits=2)

    samples = ["00"] * 50 + ["11"] * 40 + ["01"] * 6 + ["10"] * 4
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=2))
    result = runner.run_from_samples(samples)

    assert exact <= result.gs_energy + 1e-10


def test_qsci_energy_nonincreasing_with_r():
    hamiltonian = _test_hamiltonian()
    samples = ["00"] * 50 + ["11"] * 40 + ["01"] * 6 + ["10"] * 4

    e_r2 = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=2)).run_from_samples(samples).gs_energy
    e_r3 = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=3)).run_from_samples(samples).gs_energy
    e_r4 = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=4)).run_from_samples(samples).gs_energy

    assert e_r3 <= e_r2 + 1e-10
    assert e_r4 <= e_r3 + 1e-10


def test_qsci_matches_exact_with_full_basis():
    hamiltonian = _test_hamiltonian()
    exact = _exact_ground_energy(hamiltonian, n_qubits=2)

    samples = ["00", "01", "10", "11"] * 10
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=4))
    result = runner.run_from_samples(samples)

    assert result.subspace_dimension == 4
    assert result.gs_energy == pytest.approx(exact)


def test_qsci_returns_requested_number_of_roots():
    hamiltonian = _test_hamiltonian()
    exact_energies = _exact_energies(hamiltonian, n_qubits=2)

    samples = ["00", "01", "10", "11"] * 10
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=4, n_roots=2))
    result = runner.run_from_samples(samples)

    assert result.n_roots == 2
    assert result.energies.shape == (2,)
    assert result.eigenvectors.shape == (4, 2)
    assert np.allclose(result.energies, exact_energies[:2])


def test_qsci_result_ground_state_accessors():
    hamiltonian = _test_hamiltonian()
    samples = ["00", "01", "10", "11"] * 8
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=4, n_roots=3))
    result = runner.run_from_samples(samples)

    assert result.gs_energy == pytest.approx(result.energies[0])
    assert np.allclose(result.gs_coefficients, result.eigenvectors[:, 0])

    with pytest.raises(AttributeError):
        _ = result.energy
    with pytest.raises(AttributeError):
        _ = result.coefficients


def test_qsci_result_root_access_and_coefficient_map():
    hamiltonian = _test_hamiltonian()
    samples = ["00", "01", "10", "11"] * 8
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=4, n_roots=2))
    result = runner.run_from_samples(samples)

    energy_1, coeffs_1 = result.get_root(1)
    assert energy_1 == pytest.approx(result.energies[1])
    assert np.allclose(coeffs_1, result.eigenvectors[:, 1])

    coeffs_map = result.coefficients_dict(0)
    assert set(coeffs_map) == set(result.selected_bitstrings)
    assert np.allclose(np.array([coeffs_map[b] for b in result.selected_bitstrings]), result.eigenvectors[:, 0])

    with pytest.raises(IndexError):
        result.get_root(2)


def test_qsci_result_degenerate_groups():
    hamiltonian = QubitOperator("Z0")
    samples = ["00", "01", "10", "11"] * 6
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=4, n_roots=4))
    result = runner.run_from_samples(samples)

    assert np.allclose(result.energies, [-1.0, -1.0, 1.0, 1.0])
    assert result.degenerate_groups() == [[0, 1], [2, 3]]


def test_run_from_circuit_matches_samples_for_basis_state():
    hamiltonian = _test_hamiltonian()
    config = QSCIConfig(r=1, n_shots=128)
    runner = QSCI(hamiltonian, n_qubits=2, config=config)

    circuit = models.Circuit(2)
    circuit.add(gates.X(0))

    circuit_result = runner.run_from_circuit(circuit)
    sample_result = runner.run_from_samples(["10"] * 128)

    assert circuit_result.selected_bitstrings == ["10"]
    assert circuit_result.gs_energy == pytest.approx(sample_result.gs_energy)


@pytest.mark.parametrize(
    "config",
    [
        QSCIConfig(r=1),
        QSCIConfig(r=2, postselect=True, n_electrons=1),
    ],
)
def test_invalid_samples_raise(config):
    hamiltonian = _test_hamiltonian()
    runner = QSCI(hamiltonian, n_qubits=2, config=config)

    with pytest.raises(ValueError):
        runner.run_from_samples([])

    with pytest.raises(ValueError):
        runner.run_from_samples(["0"])

    with pytest.raises(ValueError):
        runner.run_from_samples(["0a"])


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        QSCIConfig(r=0)

    with pytest.raises(ValueError):
        QSCIConfig(r=2, n_roots=0)

    with pytest.raises(ValueError):
        QSCIConfig(r=2, n_roots=3)

    with pytest.raises(ValueError):
        QSCIConfig(r=1, postselect=True)

    with pytest.raises(ValueError):
        QSCIConfig(r=1, n_alpha=1)

    with pytest.raises(ValueError):
        QSCIConfig(r=1, n_electrons=1, n_alpha=1, n_beta=1)
