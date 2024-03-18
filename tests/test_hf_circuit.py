"""
Test HF reference circuit ansatz
"""

import pytest

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.driver.molecule import Molecule
from qibochem.measurement.expectation import expectation


@pytest.mark.parametrize(
    "mapping,",
    [
        None,  # JW mapping
        "bk",  # BK mapping
    ],
)
def test_h2(mapping):
    """Tests the HF circuit for H2"""
    # Hardcoded benchmark results
    h2_ref_energy = -1.117349035

    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    hamiltonian = h2.hamiltonian(ferm_qubit_map=mapping)
    circuit = hf_circuit(h2.nso, h2.nelec, ferm_qubit_map=mapping)
    hf_energy = expectation(circuit, hamiltonian)

    # assert h2.e_hf == pytest.approx(hf_energy)
    assert pytest.approx(hf_energy) == h2_ref_energy


def test_mapping_error():
    """Tests the HF circuit with an incorrect mapping"""
    with pytest.raises(KeyError):
        hf_circuit(4, 2, ferm_qubit_map="incorrect")
