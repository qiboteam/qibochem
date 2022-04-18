import pytest
from qibochem.driver import psi4_driver

def psi4_driver_test():

    geometry = """0 1 \n H 0.0 0.0 0.0 \n H 0.0 0.0 0.75 \n symmetry c1 \n units angstrom"""
    opts = {'basis': 'sto-3g', 'maxiter': 1000, 'reference': 'rhf', 'guess': 'core'}

    h2 = psi4_driver(geometry, opts)

    psi4_driver.run(h2)
    
    assert h2.energy == pytest.approx(-1.116175386322)
    
