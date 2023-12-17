"""
Test Molecule class functions
"""
import os.path

import numpy as np
import pytest
from qibo import gates, models
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from qibochem.driver import hamiltonian
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation


def test_run_pyscf():
    """PySCF driver"""
    # Hardcoded benchmark results
    # Change to run PySCF directly during a test?
    h2_ref_energy = -1.117349035
    h2_ref_hcore = np.array([[-1.14765024, -1.00692423], [-1.00692423, -1.14765024]])

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    assert h2.e_hf == pytest.approx(h2_ref_energy)
    assert np.allclose(h2.hcore, h2_ref_hcore)


def test_run_pyscf_molecule_xyz():
    """Pyscf driver with xyz file"""
    path = "./tests/lih.xyz"
    check_file = os.path.exists(path)
    if check_file == False:
        with open("./tests/lih.xyz", "a") as file:
            file.write("2\n 0 1\n Li 0.0 0.0 0.0\n H 0.0 0.0 1.2")
    lih_ref_energy = -7.83561582555692
    lih = Molecule(xyz_file="./tests/lih.xyz")
    lih.run_pyscf()

    assert lih.e_hf == pytest.approx(lih_ref_energy)


def test_hf_embedding_1():
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2))])
    mol.run_pyscf()
    ref_oei = mol.oei
    ref_tei = mol.tei
    mol.hf_embedding()
    embed_oei = mol.embed_oei
    embed_tei = mol.embed_tei
    assert np.allclose(embed_oei, ref_oei)
    assert np.allclose(embed_tei, ref_tei)


def test_hf_embedding_2():
    mol = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2))])
    mol.run_pyscf()
    mol.frozen = [0]
    mol.active = [1, 2]
    mol.hf_embedding()
    assert mol.n_active_orbs == 4
    assert mol.n_active_e == 2


def test_fermionic_hamiltonian():
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    h2_ferm_ham = hamiltonian.fermionic_hamiltonian(h2.oei, h2.tei, h2.e_nuc)

    # a^\dagger_0 a_0
    ref_one_body_tensor = np.array(
        [
            [-1.27785301, 0.0, 0.0, 0.0],
            [0.0, -1.27785301, 0.0, 0.0],
            [0.0, 0.0, -0.4482997, 0.0],
            [0.0, 0.0, 0.0, -0.4482997],
        ]
    )

    assert np.isclose(h2_ferm_ham[()], 0.7559674441714287)
    assert np.allclose(h2_ferm_ham.one_body_tensor, ref_one_body_tensor)


def test_parse_pauli_string_1():
    pauli_string = ((0, "X"), (1, "Y"))
    qibo_pauli_string = hamiltonian.parse_pauli_string(pauli_string, 0.5)
    ref_pauli_string = "0.5*X0*Y1"
    assert str(qibo_pauli_string) == ref_pauli_string


def test_parse_pauli_string_2():
    qibo_pauli_string = hamiltonian.parse_pauli_string(None, 0.1)
    assert str(qibo_pauli_string) == "0.1"


def test_qubit_hamiltonian():
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    h2_ferm_ham = hamiltonian.fermionic_hamiltonian(h2.oei, h2.tei, h2.e_nuc)
    h2_qubit_ham_jw = hamiltonian.qubit_hamiltonian(h2_ferm_ham, "jw")
    h2_qubit_ham_bk = hamiltonian.qubit_hamiltonian(h2_ferm_ham, "bk")
    ref_h2_qubit_ham_jw = {
        (): -0.04207897647782238,
        ((0, "Z"),): 0.17771287465139918,
        ((1, "Z"),): 0.1777128746513992,
        ((2, "Z"),): -0.24274280513140478,
        ((3, "Z"),): -0.24274280513140478,
        ((0, "Z"), (1, "Z")): 0.17059738328801052,
        ((0, "Z"), (2, "Z")): 0.12293305056183809,
        ((0, "Z"), (3, "Z")): 0.16768319457718972,
        ((1, "Z"), (2, "Z")): 0.16768319457718972,
        ((1, "Z"), (3, "Z")): 0.12293305056183809,
        ((2, "Z"), (3, "Z")): 0.17627640804319608,
        ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): -0.04475014401535165,
        ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): 0.04475014401535165,
        ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): 0.04475014401535165,
        ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): -0.04475014401535165,
    }
    ref_h2_qubit_ham_bk = {
        ((0, "Z"),): 0.17771287465139923,
        (): -0.04207897647782244,
        ((0, "Z"), (1, "Z")): 0.17771287465139918,
        ((2, "Z"),): -0.24274280513140484,
        ((1, "Z"), (2, "Z"), (3, "Z")): -0.24274280513140484,
        ((0, "Y"), (1, "Z"), (2, "Y")): 0.04475014401535165,
        ((0, "X"), (1, "Z"), (2, "X")): 0.04475014401535165,
        ((0, "X"), (1, "Z"), (2, "X"), (3, "Z")): 0.04475014401535165,
        ((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")): 0.04475014401535165,
        ((1, "Z"),): 0.17059738328801052,
        ((0, "Z"), (2, "Z")): 0.12293305056183809,
        ((0, "Z"), (1, "Z"), (2, "Z")): 0.16768319457718972,
        ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")): 0.16768319457718972,
        ((0, "Z"), (2, "Z"), (3, "Z")): 0.12293305056183809,
        ((1, "Z"), (3, "Z")): 0.17627640804319608,
    }

    keylist_jw = h2_qubit_ham_jw.terms.keys()
    keylist_bk = h2_qubit_ham_bk.terms.keys()
    jw_array = np.array([h2_qubit_ham_jw.terms[key] for key in keylist_jw])
    bk_array = np.array([h2_qubit_ham_bk.terms[key] for key in keylist_bk])

    keylist_ref_jw = ref_h2_qubit_ham_jw.keys()
    keylist_ref_bk = ref_h2_qubit_ham_bk.keys()
    ref_jw_array = np.array([ref_h2_qubit_ham_jw[key] for key in keylist_ref_jw])
    ref_bk_array = np.array([ref_h2_qubit_ham_bk[key] for key in keylist_ref_bk])

    assert np.allclose(jw_array, ref_jw_array)
    assert np.allclose(bk_array, ref_bk_array)


def test_symbolic_hamiltonian():
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    h2_ferm_ham = hamiltonian.fermionic_hamiltonian(h2.oei, h2.tei, h2.e_nuc)
    h2_qubit_ham = hamiltonian.qubit_hamiltonian(h2_ferm_ham, "jw")
    h2_sym_ham = hamiltonian.symbolic_hamiltonian(h2_qubit_ham)
    ref_ham = (
        -0.0420789764778224
        - 0.0447501440153516 * X(0) * X(1) * Y(2) * Y(3)
        + 0.0447501440153516 * X(0) * Y(1) * Y(2) * X(3)
        + 0.0447501440153516 * Y(0) * X(1) * X(2) * Y(3)
        - 0.0447501440153516 * Y(0) * Y(1) * X(2) * X(3)
        + 0.177712874651399 * Z(0)
        + 0.170597383288011 * Z(0) * Z(1)
        + 0.122933050561838 * Z(0) * Z(2)
        + 0.16768319457719 * Z(0) * Z(3)
        + 0.177712874651399 * Z(1)
        + 0.16768319457719 * Z(1) * Z(2)
        + 0.122933050561838 * Z(1) * Z(3)
        - 0.242742805131405 * Z(2)
        + 0.176276408043196 * Z(2) * Z(3)
        - 0.242742805131405 * Z(3)
    )
    ref_sym_ham = SymbolicHamiltonian(ref_ham)

    assert np.allclose(h2_sym_ham.matrix, ref_sym_ham.matrix)


@pytest.mark.skip(reason="psi4 doesn't offer pip install, so needs to be installed through conda or manually.")
def test_run_psi4():
    """PSI4 driver"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035
    h2_ref_hcore = np.array([[-1.14765024, -1.00692423], [-1.00692423, -1.14765024]])

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_psi4()

    assert h2.e_hf == pytest.approx(h2_ref_energy)
    assert np.allclose(h2.hcore, h2_ref_hcore)


def test_expectation_value():
    """Tests generation of molecular Hamiltonian and its expectation value using a JW-HF circuit"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    try:
        h2.run_pyscf()
    except ModuleNotFoundError:
        h2.run_psi4()

    # JW-HF circuit
    circuit = models.Circuit(h2.nso)
    circuit.add(gates.X(_i) for _i in range(sum(h2.nelec)))
    # Molecular Hamiltonian and the HF expectation value
    hamiltonian = h2.hamiltonian()
    hf_energy = expectation(circuit, hamiltonian)

    # assert h2.e_hf == pytest.approx(hf_energy)
    assert h2_ref_energy == pytest.approx(hf_energy)
