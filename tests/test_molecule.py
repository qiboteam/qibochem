"""
Test Molecule class functions
"""

from pathlib import Path

import numpy as np
import openfermion
import pytest
from qibo import gates, models
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z

from qibochem.driver import Molecule
from qibochem.measurement.expectation import expectation


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


def test_fermionic_hamiltonian():
    # Reference result
    fermion_operator_list = [
        ("", 0.7559674441714287),
        ("0^ 0", -1.277853006156875),
        ("0^ 0^ 0 0", 0.34119476657602105),
        ("0^ 0^ 2 2", 0.08950028803070331),
        ("0^ 1^ 1 0", 0.34119476657602105),
        ("0^ 1^ 3 2", 0.08950028803070331),
        ("0^ 2^ 0 2", 0.08950028803070331),
        ("0^ 2^ 2 0", 0.33536638915437944),
        ("0^ 3^ 1 2", 0.08950028803070331),
        ("0^ 3^ 3 0", 0.33536638915437944),
        ("1^ 0^ 0 1", 0.34119476657602105),
        ("1^ 0^ 2 3", 0.08950028803070331),
        ("1^ 1", -1.277853006156875),
        ("1^ 1^ 1 1", 0.34119476657602105),
        ("1^ 1^ 3 3", 0.08950028803070331),
        ("1^ 2^ 0 3", 0.08950028803070331),
        ("1^ 2^ 2 1", 0.33536638915437944),
        ("1^ 3^ 1 3", 0.08950028803070331),
        ("1^ 3^ 3 1", 0.33536638915437944),
        ("2^ 0^ 0 2", 0.3353663891543795),
        ("2^ 0^ 2 0", 0.08950028803070331),
        ("2^ 1^ 1 2", 0.3353663891543795),
        ("2^ 1^ 3 0", 0.08950028803070331),
        ("2^ 2", -0.448299696101638),
        ("2^ 2^ 0 0", 0.08950028803070331),
        ("2^ 2^ 2 2", 0.35255281608639233),
        ("2^ 3^ 1 0", 0.08950028803070331),
        ("2^ 3^ 3 2", 0.35255281608639233),
        ("3^ 0^ 0 3", 0.3353663891543795),
        ("3^ 0^ 2 1", 0.08950028803070331),
        ("3^ 1^ 1 3", 0.3353663891543795),
        ("3^ 1^ 3 1", 0.08950028803070331),
        ("3^ 2^ 0 1", 0.08950028803070331),
        ("3^ 2^ 2 3", 0.35255281608639233),
        ("3^ 3", -0.448299696101638),
        ("3^ 3^ 1 1", 0.08950028803070331),
        ("3^ 3^ 3 3", 0.35255281608639233),
    ]
    ref_h2_ferm_ham = sum(openfermion.FermionOperator(_op[0], _op[1]) for _op in fermion_operator_list)
    # Test case
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    h2_ferm_ham = h2.hamiltonian("f")

    assert h2_ferm_ham.isclose(ref_h2_ferm_ham)


@pytest.mark.parametrize(
    "mapping,expected_operators",
    [
        (
            None,
            [
                ((), -0.04207897647782238),
                (((0, "Z")), 0.17771287465139918),
                (((1, "Z")), 0.1777128746513992),
                (((2, "Z")), -0.24274280513140478),
                (((3, "Z")), -0.24274280513140478),
                (((0, "Z"), (1, "Z")), 0.17059738328801052),
                (((0, "Z"), (2, "Z")), 0.12293305056183809),
                (((0, "Z"), (3, "Z")), 0.16768319457718972),
                (((1, "Z"), (2, "Z")), 0.16768319457718972),
                (((1, "Z"), (3, "Z")), 0.12293305056183809),
                (((2, "Z"), (3, "Z")), 0.17627640804319608),
                (((0, "X"), (1, "X"), (2, "Y"), (3, "Y")), -0.04475014401535165),
                (((0, "X"), (1, "Y"), (2, "Y"), (3, "X")), 0.04475014401535165),
                (((0, "Y"), (1, "X"), (2, "X"), (3, "Y")), 0.04475014401535165),
                (((0, "Y"), (1, "Y"), (2, "X"), (3, "X")), -0.04475014401535165),
            ],
        ),  # H2 JW mapping
        (
            "bk",
            [
                ((), -0.04207897647782244),
                (((0, "Z"),), 0.17771287465139923),
                (((0, "Z"), (1, "Z")), 0.17771287465139918),
                (((2, "Z"),), -0.24274280513140484),
                (((1, "Z"), (2, "Z"), (3, "Z")), -0.24274280513140484),
                (((0, "Y"), (1, "Z"), (2, "Y")), 0.04475014401535165),
                (((0, "X"), (1, "Z"), (2, "X")), 0.04475014401535165),
                (((0, "X"), (1, "Z"), (2, "X"), (3, "Z")), 0.04475014401535165),
                (((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")), 0.04475014401535165),
                (((1, "Z"),), 0.17059738328801052),
                (((0, "Z"), (2, "Z")), 0.12293305056183809),
                (((0, "Z"), (1, "Z"), (2, "Z")), 0.16768319457718972),
                (((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")), 0.16768319457718972),
                (((0, "Z"), (2, "Z"), (3, "Z")), 0.12293305056183809),
                (((1, "Z"), (3, "Z")), 0.17627640804319608),
            ],
        ),  # H2 BK mapping
    ],
)
def test_qubit_hamiltonian(mapping, expected_operators):
    control = sum(openfermion.QubitOperator(_op, coeff) for _op, coeff in expected_operators)

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()

    h2_qubit_hamiltonian = h2.hamiltonian("q", ferm_qubit_map=mapping)
    assert h2_qubit_hamiltonian.isclose(control)


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
