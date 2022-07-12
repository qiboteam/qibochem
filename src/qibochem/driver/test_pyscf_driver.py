import sys
import pytest

sys.path.append(".")
import pyscf_driver
import numpy as np

def test_pyscf_driver():

    atoms = '''H 0.0 0.0 0.0; H 0.0 0.0 0.7122'''
    h2 = pyscf_driver.pyscf_molecule(atoms)
    pyscf_driver.pyscf_molecule.run(h2)

    h2_ref_energy = -1.1175058842
    h2_ref_hcore = np.array([[-1.13935702, -0.99230486],[-0.99230486, -1.13935702]])
    assert h2.energy == pytest.approx(h2_ref_energy)
    assert np.allclose(h2.hcore, h2_ref_hcore) 
    
