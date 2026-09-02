"""
Qibochem also has a few utility functions to assist with the construction of circuit ansatzes.
"""

from collections.abc import Sequence

import numpy as np
from qibo.config import raise_error


def generate_excitations(
    order: int, excite_from: Sequence[int], excite_to: Sequence[int], conserve_spin: bool = True
) -> list[list[int]]:
    """
    Generate all possible excitations between a list of occupied and virtual orbitals

    Args:
        order (int): Order of excitations, i.e. 1 == single, 2 == double
        excite_from (Sequence[int]): Occupied orbitals to excite from
        excite_to (Sequence[int]): Virtual orbitals to excite to
        conserve_spin (bool, optional): Whether total electronic spin is conserved when generating excitations

    Return:
        list[list[int]]: Generated excitations
    """
    # If order of excitation > either number of electrons/orbitals, return list of empty list
    if order > min(len(excite_from), len(excite_to)):
        return [[]]

    # Generate all possible excitations first
    from itertools import combinations  # pylint: disable=C0415

    all_excitations = [
        [*_from, *_to] for _from in combinations(excite_from, order) for _to in combinations(excite_to, order)
    ]
    # Filter out the excitations if conserve_spin set
    if conserve_spin:
        # Not sure if this filtering is exhaustive; might not remove some redundant excitations?
        all_excitations = [
            _ex
            for _ex in all_excitations
            if sum(_ex) % 2 == 0 and (sum(_i % 2 for _i in _ex[:order]) == sum(_i % 2 for _i in _ex[order:]))
        ]
    return _sort_excitations(all_excitations)


def _sort_excitations(excitations: list[[list[int]]]) -> list[[list[int]]]:
    """
    Sorts excitations according to some common-sense and empirical rules (see below). Order of the excitations must be
    the same throughout.

    Sorting order:
    1. (For double excitations only) All paired excitations between the same MOs first
    2. Pair up excitations between the same MOs, e.g. (0, 2) and (1, 3)
    3. Then count upwards from smallest MO
    """
    # Check that all excitations are of the same order and <= 2
    order = len(excitations[0]) // 2
    if order > 2:
        raise_error(NotImplementedError, "Can only handle single and double excitations")
    if not all(len(_ex) // 2 == order for _ex in excitations):
        raise_error(ValueError, "Cannot handle excitations of different orders")

    def sorting_fn(x):
        """Sorting function for double excitations; default sort is OK for single excitations"""
        return sum((order + 1 - _i) * abs(x[2 * _i + 1] // 2 - x[2 * _i] // 2) for _i in range(0, order))

    copy_excitations = [list(_ex) for _ex in excitations]
    result = []
    prev = []
    # Make a copy of the list of excitations, and use it populate a new list iteratively
    while copy_excitations:
        if not prev:
            # Take out all pair excitations first
            pair_excitations = [
                _ex
                for _ex in copy_excitations
                # Indices of the electrons/holes must be consecutive numbers
                if sum(abs(_ex[2 * _i + 1] // 2 - _ex[2 * _i] // 2) for _i in range(0, order)) == 0
            ]
            while pair_excitations:
                pair_excitations = sorted(pair_excitations)
                ex_to_remove = pair_excitations.pop(0)
                if ex_to_remove in copy_excitations:
                    # 'Move' the first pair excitation from copy_excitations to result
                    index = copy_excitations.index(ex_to_remove)
                    result.append(copy_excitations.pop(index))

            # No more pair excitations, only remaining excitations should have >=3 MOs involved
            # Sort the remaining excitations
            copy_excitations = sorted(copy_excitations, key=sorting_fn if order != 1 else None)
        else:
            # Check to see for excitations involving the same MOs as prev
            _from = prev[:order]
            _to = prev[order:]
            # Get all possible excitations involving the same MOs
            new_from = [_i + 1 if _i % 2 == 0 else _i - 1 for _i in _from]
            new_to = [_i + 1 if _i % 2 == 0 else _i - 1 for _i in _to]
            same_mo_ex = [sorted(list(_f) + list(_t)) for _f in (_from, new_from) for _t in (_to, new_to)]
            # Remove the excitations with the same MOs from copy_excitations
            while same_mo_ex:
                same_mo_ex = sorted(same_mo_ex)
                ex_to_remove = same_mo_ex.pop(0)
                if ex_to_remove in copy_excitations:
                    # 'Move' the first entry of same_mo_index from copy_excitations to result
                    index = copy_excitations.index(ex_to_remove)
                    result.append(copy_excitations.pop(index))
            prev = None
            continue
        if copy_excitations:
            # Remove the first entry from the sorted list of remaining excitations and add it to result
            prev = copy_excitations.pop(0)
            result.append(prev)
    return result


def mp2_amplitude(excitation: Sequence[int], orbital_energies: Sequence[float], tei: np.ndarray) -> float:
    r"""
    Calculate MP2 guess amplitude for a fermionic excitation. Single excitation will be: 0.0, a double excitation
    (In SO basis): :math:`t_{ij}^{ab} = (g_{ijab} - g_{ijba}) / (e_i + e_j - e_a - e_b)`

    Args:
        excitation (Sequence[int]): Orbitals involved in the excitation. Must have either 2 or 4 elements, representing
            a single or double excitation respectively
        orbital_energies (Sequence[float]): Eigenvalues of the Fock operator, i.e. orbital energies
        tei (np.ndarray): Two-electron integrals in MO basis and second quantization notation

    Returns:
        float: MP2 guess amplitude
    """
    # Check validity of excitation argument
    if len(excitation) not in (2, 4):
        raise_error(ValueError, f"{excitation} must have either 2 or 4 elements")
    # Single excitation => Can just return 0.0 directly
    if len(excitation) == 2:
        return 0.0

    # Convert orbital indices to be in MO basis
    mo_orbitals = [orbital // 2 for orbital in excitation]
    # Numerator: g_ijab - g_ijba
    g_ijab = (
        tei[tuple(mo_orbitals)]  # Can index directly using the MO TEIs
        if (excitation[0] + excitation[3]) % 2 == 0 and (excitation[1] + excitation[2]) % 2 == 0
        else 0.0
    )
    g_ijba = (
        tei[tuple(mo_orbitals[:2] + mo_orbitals[2:][::-1])]  # Reverse last two terms
        if (excitation[0] + excitation[2]) % 2 == 0 and (excitation[1] + excitation[3]) % 2 == 0
        else 0.0
    )
    numerator = g_ijab - g_ijba
    # Denominator is directly from the orbital energies
    denominator = sum(orbital_energies[mo_orbitals[:2]]) - sum(orbital_energies[mo_orbitals[2:]])
    return numerator / denominator
