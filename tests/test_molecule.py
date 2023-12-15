"""
Test Molecule class functions
"""
import os.path
import numpy as np
import pytest
from qibo import gates, models

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
    path = './lih.xyz'
    check_file = os.path.exists(path)
    if check_file == False:
        with open("lih.xyz", "a") as file:
            file.write("2\n 0 1\n Li 0.0 0.0 0.0\n H 0.0 0.0 1.2")
    lih_ref_energy = -7.83561582555692
    lih = Molecule(xyz_file="lih.xyz")
    lih.run_pyscf()

    assert lih.e_hf == pytest.approx(lih_ref_energy)


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
