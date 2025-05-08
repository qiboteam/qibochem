import numpy as np
import pytest
from qibo import symbols
from qibo.models import Circuit
from qibo.hamiltonians import SymbolicHamiltonian
from qibochem.measurement import expectation

from qibochem.driver.hamiltonian import _symbolic_identity, build_folded_hamiltonian


def test_symbolic_identity_basic():
    # For 1 qubit, should be I(0)
    expr = _symbolic_identity(1)
    assert str(expr) == str(symbols.I(0))
    # For 2 qubits, should be I(0)*I(1)
    expr2 = _symbolic_identity(2)
    assert str(expr2) == str(symbols.I(0) * symbols.I(1))
    # For 0 qubits, should be 1
    expr0 = _symbolic_identity(0)
    assert expr0 == 1


def test_build_folded_hamiltonian_symbolic():
    # H = 0.5*Z0 + 0.3*X1, nqubits=2
    nqubits = 2
    H = SymbolicHamiltonian(0.5 * symbols.Z(0) + 0.3 * symbols.X(1), nqubits)
    lam = 1.1
    folded = build_folded_hamiltonian(H, lam)
    # Should be a SymbolicHamiltonian
    assert isinstance(folded, SymbolicHamiltonian)
    # Should have correct number of qubits
    assert folded.nqubits == nqubits
    # Check that the folded Hamiltonian is (H - lam*I)^2
    # Evaluate on the |00> state

    c = Circuit(nqubits)
    folded_val = expectation(c, folded)
    # Compute expected value manually
    # For |00>, <Z0>=1, <X1>=0, <I>=1
    expected = 0.5**2 + 0.3**2 + lam**2 - lam
    assert np.isclose(folded_val, expected)


def test_build_folded_hamiltonian_qubitop():
    # Test with OpenFermion QubitOperator input
    try:
        from openfermion import QubitOperator
    except ImportError:
        pytest.skip("openfermion not installed")
    nqubits = 1
    Hq = QubitOperator("Z0", 0.7)
    lam = 0.2
    folded = build_folded_hamiltonian(Hq, lam)
    assert isinstance(folded, SymbolicHamiltonian)
    # Should match (0.7*<Z0> - 0.2)^2 on |0>

    c = Circuit(nqubits)
    val = expectation(c, folded)
    expected = (0.7 * 1 - 0.2) ** 2
    assert np.isclose(val, expected)
