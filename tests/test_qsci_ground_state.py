"""Tests for ground-state QSCI."""

import numpy as np
import pytest
from openfermion import QubitOperator, get_sparse_operator
from qibo import gates, models

from qibochem.selected_ci.qsci import (
    QSCI,
    QSCIConfig,
    _apply_pauli_term,
    _project_hamiltonian_subspace,
    _validate_and_copy_counts,
    qsci_ground_state,
)


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


def test_projected_matrix_warns_for_nonhermitian_input_and_symmetrizes():
    hamiltonian = (1.0 + 1.0j) * QubitOperator("X0")

    with pytest.warns(UserWarning, match="not Hermitian"):
        projected = _project_hamiltonian_subspace(hamiltonian, ["0", "1"])

    assert np.allclose(projected, projected.conj().T)
    assert np.allclose(projected, np.array([[0.0, 1.0], [1.0, 0.0]]))


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


def test_run_from_circuit_warns_on_extra_qubits_and_continues():
    hamiltonian = _test_hamiltonian()
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=1, n_shots=64, seed=7))

    circuit = models.Circuit(3)
    circuit.add(gates.X(2))

    with pytest.warns(UserWarning, match="more qubits"):
        result = runner.run_from_circuit(circuit)

    assert result.n_shots_used == 64
    assert all(len(bitstring) == 2 for bitstring in result.selected_bitstrings)


def test_run_from_circuit_raises_when_circuit_has_fewer_qubits():
    hamiltonian = _test_hamiltonian()
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=1, n_shots=16))

    circuit = models.Circuit(1)
    with pytest.raises(ValueError, match="fewer qubits"):
        runner.run_from_circuit(circuit)


def test_run_from_circuit_requires_n_shots_when_not_configured():
    hamiltonian = _test_hamiltonian()
    runner = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=1))

    with pytest.raises(ValueError, match="n_shots"):
        runner.run_from_circuit(models.Circuit(2))


def test_run_from_circuit_seeded_sampling_is_reproducible():
    hamiltonian = QubitOperator("Z0")
    runner = QSCI(hamiltonian, n_qubits=1, config=QSCIConfig(r=2, n_shots=256, seed=123))

    circuit = models.Circuit(1)
    circuit.add(gates.H(0))

    result_1 = runner.run_from_circuit(circuit)
    result_2 = runner.run_from_circuit(circuit)

    assert result_1.counts == result_2.counts
    assert result_1.selected_bitstrings == result_2.selected_bitstrings


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

    with pytest.raises(ValueError):
        QSCIConfig(r=1, n_shots=0)

    with pytest.raises(ValueError):
        QSCIConfig(r=1, n_electrons=-1)

    with pytest.raises(ValueError):
        QSCIConfig(r=1, n_alpha=-1, n_beta=0)

    with pytest.raises(ValueError):
        QSCIConfig(r=1, n_alpha=0, n_beta=-1)


def test_qsci_constructor_validations_raise():
    with pytest.raises(TypeError):
        QSCI("not-a-qubit-operator", n_qubits=2, config=QSCIConfig(r=1))

    with pytest.raises(ValueError):
        QSCI(QubitOperator("Z0"), n_qubits=0, config=QSCIConfig(r=1))


def test_postselect_can_empty_counts_and_raise():
    hamiltonian = _test_hamiltonian()
    runner = QSCI(
        hamiltonian,
        n_qubits=2,
        config=QSCIConfig(r=2, postselect=True, n_electrons=2),
    )

    with pytest.raises(ValueError, match="No valid bitstrings"):
        runner.run_from_samples(["00", "01", "10"])


def test_spin_resolved_postselection_works():
    hamiltonian = QubitOperator("Z0") + QubitOperator("Z1")
    runner = QSCI(
        hamiltonian,
        n_qubits=4,
        config=QSCIConfig(r=4, postselect=True, n_alpha=1, n_beta=1),
    )

    result = runner.run_from_samples(["1100", "1010", "0101", "0110", "0011"])
    assert set(result.selected_bitstrings) == {"1100", "0110", "0011"}
    assert result.subspace_dimension == 3


def test_validate_counts_truncates_and_aggregates_when_enabled():
    with pytest.warns(UserWarning, match="truncating"):
        counts = _validate_and_copy_counts({"100": 2, "101": 3, "01": 4}, 2, allow_truncation=True)

    assert counts == {"10": 5, "01": 4}


def test_validate_counts_ignores_nonpositive_values():
    counts = _validate_and_copy_counts({"00": 0, "01": -2, "10": 5}, 2)
    assert counts == {"10": 5}


def test_apply_pauli_term_invalid_operator_raises():
    with pytest.raises(ValueError, match="Unsupported Pauli operator"):
        _apply_pauli_term(((0, "A"),), "0")


def test_degenerate_groups_single_root_returns_empty_list():
    hamiltonian = _test_hamiltonian()
    result = QSCI(hamiltonian, n_qubits=2, config=QSCIConfig(r=1, n_roots=1)).run_from_samples(["00"] * 16)
    assert result.degenerate_groups() == []


def test_qsci_ground_state_wrapper_samples_and_circuit_paths():
    hamiltonian = _test_hamiltonian()

    result_samples = qsci_ground_state(
        hamiltonian,
        n_qubits=2,
        samples=["00", "00", "11", "01"],
        r=2,
    )
    assert result_samples.subspace_dimension == 2

    circuit = models.Circuit(2)
    circuit.add(gates.X(0))
    result_circuit = qsci_ground_state(
        hamiltonian,
        n_qubits=2,
        circuit=circuit,
        config=QSCIConfig(r=1, n_shots=64),
    )
    assert result_circuit.selected_bitstrings == ["10"]


def test_qsci_ground_state_wrapper_invalid_argument_combinations_raise():
    hamiltonian = _test_hamiltonian()
    circuit = models.Circuit(2)

    with pytest.raises(ValueError, match="either `config` or `config_kwargs`"):
        qsci_ground_state(
            hamiltonian,
            n_qubits=2,
            samples=["00"],
            config=QSCIConfig(r=1),
            r=1,
        )

    with pytest.raises(ValueError, match="exactly one"):
        qsci_ground_state(hamiltonian, n_qubits=2, r=1)

    with pytest.raises(ValueError, match="exactly one"):
        qsci_ground_state(hamiltonian, n_qubits=2, samples=["00"], circuit=circuit, r=1)
