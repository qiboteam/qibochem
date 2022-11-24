"""
Test Molecule class functions
"""

import numpy as np
import pytest

from qibochem.driver.molecule import Molecule



def test_run_pyscf():
    """PySCF driver"""
    # Hardcoded benchmark results
    # Change to run PySCF directly during a test?
    h2_ref_energy = -1.117349035
    h2_ref_hcore = np.array([[-1.14765024, -1.00692423], [-1.00692423, -1.14765024]])

    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    assert h2.e_hf == pytest.approx(h2_ref_energy)
    assert np.allclose(h2.hcore, h2_ref_hcore)


def test_run_psi4():
    """PSI4 driver"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035
    h2_ref_hcore = np.array([[-1.14765024, -1.00692423], [-1.00692423, -1.14765024]])

    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))])
    h2.run_psi4()

    assert h2.e_hf == pytest.approx(h2_ref_energy)
    assert np.allclose(h2.hcore, h2_ref_hcore)


