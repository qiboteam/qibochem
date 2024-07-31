"""
Tests for ansatz utility functions (util.py)
"""

import numpy as np
import pytest

from qibochem.ansatz.util import generate_excitations, mp2_amplitude, sort_excitations
from qibochem.driver import Molecule


@pytest.mark.parametrize(
    "order,excite_from,excite_to,expected",
    [
        (1, [0, 1], [2, 3], [[0, 2], [1, 3]]),
        (2, [2, 3], [4, 5], [[2, 3, 4, 5]]),
        (3, [0, 1], [2, 3], [[]]),
    ],
)
def test_generate_excitations(order, excite_from, excite_to, expected):
    """Test generation of all possible excitations between two lists of orbitals"""
    test = generate_excitations(order, excite_from, excite_to)
    assert test == expected


@pytest.mark.parametrize(
    "order,excite_from,excite_to,expected",
    [
        (1, [0, 1], [2, 3, 4, 5], [[0, 2], [1, 3], [0, 4], [1, 5]]),
        (2, [0, 1], [2, 3, 4, 5], [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 5], [0, 1, 3, 4]]),
    ],
)
def test_sort_excitations(order, excite_from, excite_to, expected):
    test = sort_excitations(generate_excitations(order, excite_from, excite_to))
    assert test == expected


def test_sort_excitations_triples():
    with pytest.raises(NotImplementedError):
        sort_excitations([[1, 2, 3, 4, 5, 6]])


def test_mp2_amplitude_singles():
    assert mp2_amplitude([0, 2], np.random.rand(4), np.random.rand(4, 4)) == 0.0


def test_mp2_amplitude_doubles():
    h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7))])
    h2.run_pyscf()
    l = mp2_amplitude([0, 1, 2, 3], h2.eps, h2.tei)
    ref_l = 0.06834019757197053

    assert np.isclose(l, ref_l)
