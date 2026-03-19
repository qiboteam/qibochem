"""
Test Molecule class functions
"""

from pathlib import Path

import numpy as np
import openfermion
import pytest
from qibo import gates, models
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z

from qibochem.driver import Molecule
from qibochem.measurement import expectation


@pytest.mark.parametrize(
    "xyz_file,expected",
    [
        (None, -1.117349035),
        ("lih.xyz", -7.83561582555692),
        ("h2.xyz", -1.117349035),
    ],
)
def test_pyscf_driver(xyz_file, expected):
    if xyz_file is None:
        mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    else:
        file_path = Path("tests", "data") / Path(xyz_file)
        # In case .xyz files somehow not found
        if not file_path.is_file():
            with open(file_path, "w") as file_handler:
                if xyz_file == "lih.xyz":
                    file_handler.write("2\n0 1\nLi 0.0 0.0 0.0\nH 0.0 0.0 1.2\n")
                elif xyz_file == "h2.xyz":
                    file_handler.write("2\n \nH 0.0 0.0 0.0\nH 0.0 0.0 0.7\n")
                else:
                    file_handler.write("2\n \nH 0.0 0.0 0.0\nH 0.0 0.0 0.7\n")
        # Define Molecule using .xyz file
        mol = Molecule(xyz_file=file_path)
    # Run PySCF and check that the HF energy matches
    mol.run_pyscf()
    assert mol.e_hf == pytest.approx(expected)


# Commenting out since not actively supporting PSI4 at the moment
# @pytest.mark.skip(reason="Psi4 doesn't offer pip install, so needs to be installed through conda or manually.")
# def test_run_psi4():
#     """PSI4 driver"""
#     # Hardcoded benchmark results
#     h2_ref_energy = -1.117349035
#
#     h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
#     h2.run_psi4()
#
#     assert h2.e_hf == pytest.approx(h2_ref_energy)


def test_molecule_custom_basis():
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2))], 0, 1, "6-31g")
    mol.run_pyscf()
    assert np.isclose(mol.e_hf, -7.94129296352493)


@pytest.mark.parametrize(
    "active,frozen,expected",
    [
        (None, None, (list(range(6)), [])),  # Default arguments: Nothing given
        ([1, 2, 5], None, ([1, 2, 5], [0])),  # Default frozen argument if active given
        (None, [0], (list(range(1, 6)), [0])),  # Default active argument if frozen given
        ([0, 1, 2, 3], [], (list(range(4)), [])),  # active, frozen arguments both given
        ([1, 2, 3], [0], (list(range(1, 4)), [0])),  # active, frozen arguments both given
    ],
)
def test_define_active_space(active, frozen, expected):
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2))])
    mol.nalpha = 2
    mol.norb = 6
    assert mol._active_space(active, frozen) == expected


def test_define_active_space_assertions():
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2))])
    mol.nalpha = 2
    mol.norb = 6

    # Invalid active argument
    with pytest.raises(AssertionError):
        _ = mol._active_space([10], None)
    # Invalid frozen argument
    with pytest.raises(AssertionError):
        _ = mol._active_space(None, [100])
    # active/frozen spaces overlap
    with pytest.raises(AssertionError):
        _ = mol._active_space([0, 1], [0])


def test_hf_embedding():
    mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))], active=[0])
    mol.run_pyscf()
    # Remove all virtual orbitals from the active space
    mol.hf_embedding()
    # Check that the class attributes have been updated correctly
    assert mol.frozen == []
    assert mol.n_active_orbs == 2
    assert mol.n_active_e == 2
    # OEI/TEI (in MO basis) for the occupied orbitals should remain unchanged
    dim = mol.n_active_orbs // 2
    assert np.allclose(mol.embed_oei, mol.oei[:dim, :dim])
    assert np.allclose(mol.embed_tei, mol.tei[:dim, :dim, :dim, :dim])


@pytest.mark.parametrize(
    "option,expected",
    [
        ("f", sum(openfermion.FermionOperator(f"{_i}^ {_i}", (-1) ** ((_i // 2) + 1)) for _i in range(4))),
        ("q", 0.5 * sum(openfermion.QubitOperator(f"Z{_i}", (-1) ** (_i // 2)) for _i in range(4))),
    ],
)
def test_hamiltonian(option, expected):
    """Test option to return FermionOperator/QubitOperator"""
    dummy = Molecule()
    dummy.e_nuc = 0.0
    dummy.oei = np.diag((-1.0, 1.0))
    dummy.tei = np.zeros((2, 2, 2, 2))  # Basically, only one-electron operators in the Hamiltonian

    test_ham = dummy.hamiltonian(option)
    assert test_ham.isclose(expected)


def test_hamiltonian_input_errors():
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.e_nuc = 0.0
    h2.oei = np.random.rand(4, 4)
    h2.tei = np.random.rand(4, 4, 4, 4)
    # Hamiltonian type error
    with pytest.raises(NameError):
        h2.hamiltonian("ihpc")
    # Fermion to qubit mapping error
    with pytest.raises(KeyError):
        h2.hamiltonian(ferm_qubit_map="incorrect")


def test_expectation_value():
    """Tests generation of molecular Hamiltonian and its expectation value using a JW-HF circuit"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()

    # JW-HF circuit
    circuit = models.Circuit(h2.nso)
    circuit.add(gates.X(_i) for _i in range(h2.nelec))
    # Molecular Hamiltonian and the HF expectation value
    hamiltonian = h2.hamiltonian()
    hf_energy = expectation(circuit, hamiltonian)

    # assert h2.e_hf == pytest.approx(hf_energy)
    assert h2_ref_energy == pytest.approx(hf_energy)


def test_fs_hamiltonian():
    """Test folded spectrum Hamiltonian method"""
    mol = Molecule()  # Dummy molecule
    hamiltonian = SymbolicHamiltonian(0.5 * Z(0) + 0.3 * X(1), nqubits=2)
    omega = 1.1
    folded = mol.fs_hamiltonian(omega, hamiltonian)
    # Check matrix of the folded Hamiltonian (H - omega*I)^2
    original_ham = 0.5 * np.kron(Z(0).matrix, np.eye(2)) + 0.3 * np.kron(np.eye(2), X(1).matrix)
    folded_matrix = (original_ham - omega * np.eye(4)) @ (original_ham - omega * np.eye(4))
    assert np.allclose(folded.matrix, folded_matrix)


def test_fs_hamiltonian_default():
    """Test use of molecular Hamiltonian as the default if not given"""
    dummy = Molecule()
    dummy.e_nuc = 0.0
    dummy.oei = np.diag((-1.0, 0.0))
    dummy.tei = np.zeros((2, 2, 2, 2))  # Basically, only one-electron operators in the Hamiltonian
    dummy_ham = dummy.hamiltonian()

    omega = 0.0
    folded = dummy.fs_hamiltonian(omega)
    assert np.allclose(folded.matrix, dummy_ham.matrix @ dummy_ham.matrix)


@pytest.mark.parametrize(
    "hamiltonian,n_eigvals",
    [
        (openfermion.reverse_jordan_wigner(openfermion.QubitOperator("Z0 Z1")), 2),
        (openfermion.QubitOperator("Z0 Z1"), 2),
        (SymbolicHamiltonian(Z(0) * Z(1)), None),
    ],
)
def test_eigenvalues(hamiltonian, n_eigvals):
    """Common set of eigenvalues: [-1.0, -1.0, 1.0, 1.0]"""
    dummy = Molecule()
    result = dummy.eigenvalues(hamiltonian)
    # Expected: sorted(np.kron(np.array([1.0, -1.0]), np.array([1.0, -1.0])))
    assert np.allclose(result, np.array([-1.0, -1.0, 1.0, 1.0])[:n_eigvals])


def test_eigenvalues_error():
    dummy = Molecule()
    # Unknown Hamiltonian type
    with pytest.raises(TypeError):
        dummy.eigenvalues(0.0)
