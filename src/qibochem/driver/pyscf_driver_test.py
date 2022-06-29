import pytest
from qibochem.driver import pyscf_driver

def pyscf_driver_test():

    atoms = '''H 0.0 0.0 0.0; H 0.0 0.0 0.7122'''

    h2 = pyscf_molecule(atoms)

    pyscf_molecule.run(h2)
    
    assert h2.energy == pytest.approx(-1.11750588420433)
    
