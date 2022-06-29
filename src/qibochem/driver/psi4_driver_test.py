import pytest
from qibochem.driver import psi4_driver

def psi4_driver_test():

    geometry = """0 1 \n H 0.0 0.0 0.0 \n H 0.0 0.0 0.7122 \n symmetry c1 \n units angstrom"""
    opts = {'basis': 'sto-3g', 'maxiter': 1000, 'reference': 'rhf', 'guess': 'core', 'scftype': 'direct',}

    h2 = psi4_molecule(geometry, opts)

    psi4_molecule.run(h2)
    
    assert h2.energy == pytest.approx(-1.117505884204352)
    
