"""
Test functionality to reduce the measurement cost of running VQE
"""

import pytest

# from qibochem.driver import Molecule
# from qibochem.measurement import expectation
from qibochem.measurement.optimization import (
    check_terms_commutativity,
    group_commuting_terms,
)

# from qibo import Circuit, gates
# from qibo.hamiltonians import SymbolicHamiltonian
# from qibo.symbols import X, Z


@pytest.mark.parametrize(
    "term1,term2,qwc_expected,gc_expected",
    [
        ("X0", "Z0", False, False),
        ("X0", "Z1", True, True),
        ("X0 X1", "Y0 Y1", False, True),
        ("X0 Y1", "Y0 Y1", False, False),
    ],
)
def test_check_terms_commutativity(term1, term2, qwc_expected, gc_expected):
    """Do two Pauli strings commute (qubit-wise or generally)?"""
    qwc_result = check_terms_commutativity(term1, term2, qubitwise=True)
    assert qwc_result == qwc_expected
    gc_result = check_terms_commutativity(term1, term2, qubitwise=False)
    assert gc_result == gc_expected


# @pytest.mark.parametrize(
#     "term_list,qwc_expected,gc_expected",
#     [
#         ([["X0"], ["Z0"], ["Z1"], ["Z0", "Z1"]], {[["Z0"], ["Z1"], ["Z0", "Z1"]], ["X0"]}, {[["Z0"], ["Z1"], ["Z0", "Z1"]], ["X0"]}),
#         # (["X0"], ["Z1"], True, True),
#         # (["X0", "X1"], ["Y0", "Y1"], False, True),
#         # (["X0", "Y1"], ["Y0", "Y1"], False, False),
#     ],
# )
# def test_group_commuting_terms(term_list, qwc_expected, gc_expected):
#     qwc_result = group_commuting_terms(term_list, qubitwise=True)
#     assert set(qwc_result) == qwc_expected
#     gc_result = group_commuting_terms(term_list, qubitwise=False)
#     assert set(gc_result) == gc_expected
