"""
Qibochem also has a few utility functions to assist with the construction of circuit
ansatzes.
"""

from collections.abc import Sequence

import numpy as np
from qibo.config import raise_error


def generate_excitations(
    order: int,
    excite_from: Sequence[int],
    excite_to: Sequence[int],
    conserve_spin: bool = True,
) -> list[list[int]]:
    """
    Generate all possible excitations between a list of occupied and virtual orbitals

    Args:
        order (int):
            Order of excitations, i.e. 1 == single, 2 == double
        excite_from (Sequence[int]):
            Occupied orbitals to excite from
        excite_to (Sequence[int]):
            Virtual orbitals to excite to
        conserve_spin (bool, optional):
            Whether total electronic spin is conserved when generating excitations

    Return:
        list[list[int]]: Generated excitations
    """
    # If order of excitation > either n_electrons/n_orbitals, return list of empty list
    if order > min(len(excite_from), len(excite_to)):
        return [[]]

    # Generate all possible excitations first
    from itertools import combinations  # pylint: disable=C0415

    all_excitations = [
        [*_from, *_to]
        for _from in combinations(excite_from, order)
        for _to in combinations(excite_to, order)
    ]
    # Filter out the excitations if conserve_spin set
    if conserve_spin:
        # Not sure if exhaustive; might not remove some redundant excitations?
        all_excitations = [
            _ex
            for _ex in all_excitations
            if sum(_ex) % 2 == 0
            and (sum(_i % 2 for _i in _ex[:order]) == sum(_i % 2 for _i in _ex[order:]))
        ]
    return _sort_excitations(all_excitations)


def _sort_excitations(excitations: list[list[int]]) -> list[list[int]]:
    """Sort excitations using MO pairing and ascending-MO rules."""
    order = len(excitations[0]) // 2
    if order > 2:
        raise_error(
            NotImplementedError,
            "Can only handle single and double excitations",
        )
    if not all(len(excitation) // 2 == order for excitation in excitations):
        raise_error(ValueError, "Cannot handle excitations of different orders")

    def mo_distance(excitation: list[int]) -> int:
        """Sort double excitations; default sort is OK for single excitations"""
        return sum(
            (order + 1 - index)
            * abs(excitation[2 * index + 1] // 2 - excitation[2 * index] // 2)
            for index in range(order)
        )

    def related_excitations(excitation: list[int]) -> list[list[int]]:
        """Correlate excitations between the same set of MOs"""
        occupied = excitation[:order]
        virtual = excitation[order:]

        def spin_partner(orbital: int) -> int:
            return orbital + 1 if orbital % 2 == 0 else orbital - 1

        partner_occupied = [spin_partner(orbital) for orbital in occupied]
        partner_virtual = [spin_partner(orbital) for orbital in virtual]

        return sorted(
            [
                sorted(source + target)
                for source in (occupied, partner_occupied)
                for target in (virtual, partner_virtual)
            ]
        )

    remaining = [list(excitation) for excitation in excitations]
    result = []
    previous = None
    while remaining:
        if previous is None:
            pair_excitations = sorted(
                excitation
                for excitation in remaining
                if all(
                    excitation[2 * index] // 2 == excitation[2 * index + 1] // 2
                    for index in range(order)
                )
            )
            result.extend(
                remaining.pop(remaining.index(excitation))
                for excitation in pair_excitations
            )
            remaining.sort(key=mo_distance if order != 1 else None)
            if remaining:
                previous = remaining.pop(0)
                result.append(previous)
        else:
            result.extend(
                remaining.pop(remaining.index(excitation))
                for excitation in related_excitations(previous)
                if excitation in remaining
            )
            previous = None
    return result


def mp2_amplitude(
    excitation: Sequence[int], orbital_energies: Sequence[float], tei: np.ndarray
) -> float:
    r"""
    Calculate MP2 guess amplitude for a fermionic excitation. Single excitation: 0.0;
    double excitation (In SO basis):
    :math:`t_{ij}^{ab} = (g_{ijab} - g_{ijba}) / (e_i + e_j - e_a - e_b)`

    Args:
        excitation (Sequence[int]):
            Orbitals involved in the excitation. Must have either 2 or 4 elements,
            representing a single or double excitation respectively
        orbital_energies (Sequence[float]):
            Eigenvalues of the Fock operator, i.e. orbital energies
        tei (np.ndarray):
            Two-electron integrals in MO basis and second quantization notation

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
    # Numerator term: g_ijab - g_ijba
    g_ijab = (
        tei[tuple(mo_orbitals)]  # Can index directly using the MO TEIs
        if (excitation[0] + excitation[3]) % 2 == 0
        and (excitation[1] + excitation[2]) % 2 == 0
        else 0.0
    )
    g_ijba = (
        tei[tuple(mo_orbitals[:2] + mo_orbitals[2:][::-1])]  # Reverse last two terms
        if (excitation[0] + excitation[2]) % 2 == 0
        and (excitation[1] + excitation[3]) % 2 == 0
        else 0.0
    )
    numerator = g_ijab - g_ijba
    # Denominator is directly from the orbital energies
    denominator = sum(orbital_energies[mo_orbitals[:2]]) - sum(
        orbital_energies[mo_orbitals[2:]]
    )
    return numerator / denominator
