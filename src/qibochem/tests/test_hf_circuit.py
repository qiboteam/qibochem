"""
Test HF reference circuit ansatz
"""

# import numpy as np
import pytest

from qibochem.driver.molecule import Molecule
from qibochem.ansatz.hf_reference import hf_circuit


def test_jw_circuit():
    """Tests the HF circuit with the Jordan-Wigner mapping"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))])
    try:
        h2.run_pyscf()
    except ModuleNotFoundError:
        h2.run_psi4()

    # JW-HF circuit
    circuit = hf_circuit(h2.nso, sum(h2.nelec), ferm_qubit_map=None)

    # Molecular Hamiltonian and the HF expectation value
    ferm_ham = h2.fermionic_hamiltonian()
    qubit_ham = h2.qubit_hamiltonian(ferm_ham)
    sym_ham = h2.symbolic_hamiltonian(qubit_ham)
    hf_energy = h2.expectation_value(circuit, sym_ham)

    # assert h2.e_hf == pytest.approx(hf_energy)
    assert h2_ref_energy == pytest.approx(hf_energy)


def test_bk_circuit_1():
    """Tests the HF circuit with the Brayvi-Kitaev mapping for H2"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))])
    try:
        h2.run_pyscf()
    except ModuleNotFoundError:
        h2.run_psi4()

    # JW-HF circuit
    circuit = hf_circuit(h2.nso, sum(h2.nelec), ferm_qubit_map='bk')

    # Molecular Hamiltonian and the HF expectation value
    ferm_ham = h2.fermionic_hamiltonian()
    qubit_ham = h2.qubit_hamiltonian(ferm_ham, ferm_qubit_map='bk')
    sym_ham = h2.symbolic_hamiltonian(qubit_ham)
    hf_energy = h2.expectation_value(circuit, sym_ham)

    # assert h2.e_hf == pytest.approx(hf_energy)
    assert h2_ref_energy == pytest.approx(hf_energy)


def test_bk_circuit_2():
    """Tests the HF circuit with the Brayvi-Kitaev mapping for LiH"""
    # Hardcoded benchmark results
    lih = Molecule([('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.3))])
    try:
        lih.run_pyscf()
    except ModuleNotFoundError:
        lih.run_psi4()

    # JW-HF circuit
    circuit = hf_circuit(lih.nso, sum(lih.nelec), ferm_qubit_map='bk')

    # Molecular Hamiltonian and the HF expectation value
    ferm_ham = lih.fermionic_hamiltonian()
    qubit_ham = lih.qubit_hamiltonian(ferm_ham, ferm_qubit_map='bk')
    sym_ham = lih.symbolic_hamiltonian(qubit_ham)
    hf_energy = lih.expectation_value(circuit, sym_ham)

    assert lih.e_hf == pytest.approx(hf_energy)
