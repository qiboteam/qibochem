import sys
import pytest

sys.path.append(".")
import molecule
import numpy as np

def test_molecule():

    h2 = molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))])
    h2.run_pyscf()

    h2_ref_energy = -1.1175058842
    h2_ref_hcore = np.array([[-1.13935702, -0.99230486],[-0.99230486, -1.13935702]])
    assert h2.e_hf == pytest.approx(h2_ref_energy)
    assert np.allclose(h2.hcore, h2_ref_hcore)
    
