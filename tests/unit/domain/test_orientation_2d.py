# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from itertools import combinations, permutations, product

from neuroframe.domain.orientation_2d import Orientation2D

_AXIS_GROUPS = (
    ("p", "a"),
    ("i", "s"),
    ("r", "l"),
)
_AXIS_PAIRS = tuple(frozenset(axis_group) for axis_group in _AXIS_GROUPS)



# ================================================================
# 1. Section: Contract Tests
# ================================================================
@pytest.mark.unit
def test_orientation_2d_contains_24_members() -> None:
    assert len(Orientation2D) == 24

@pytest.mark.unit
def test_orientation_2d_contains_all_and_only_valid_values() -> None:
    expected_values = _expected_orientation_values()
    actual_values = {member.value for member in Orientation2D}

    assert actual_values == expected_values

@pytest.mark.unit
def test_orientation_2d_has_no_duplicate_values() -> None:
    values = [member.value for member in Orientation2D]

    assert len(values) == len(set(values))

@pytest.mark.unit
@pytest.mark.parametrize("orientation", list(Orientation2D), ids=lambda item: item.name)
def test_orientation_2d_names_are_uppercase_and_values_are_lowercase(
    orientation: Orientation2D,
) -> None:
    assert orientation.name == orientation.name.upper()
    assert orientation.value == orientation.value.lower()
    assert orientation.name == orientation.value.upper()

@pytest.mark.unit
@pytest.mark.parametrize("orientation", list(Orientation2D), ids=lambda item: item.name)
def test_orientation_2d_values_have_two_unique_letters(
    orientation: Orientation2D,
) -> None:
    value = orientation.value

    assert len(value) == 2
    assert len(set(value)) == 2

@pytest.mark.unit
@pytest.mark.parametrize("orientation", list(Orientation2D), ids=lambda item: item.name)
def test_orientation_2d_values_contain_letters_from_two_distinct_axes(
    orientation: Orientation2D,
) -> None:
    assert _contains_exactly_one_letter_from_two_distinct_axes(orientation.value)

@pytest.mark.unit
def test_orientation_2d_order_is_deterministic_and_sorted_by_name() -> None:
    names = [member.name for member in Orientation2D]

    assert names == sorted(names)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _expected_orientation_values() -> set[str]:
    expected: set[str] = set()

    for selected_axis_groups in combinations(_AXIS_GROUPS, 2):
        for selected_letters in product(*selected_axis_groups):
            for ordered_letters in permutations(selected_letters):
                expected.add("".join(ordered_letters))

    return expected

def _contains_exactly_one_letter_from_two_distinct_axes(code: str) -> bool:
    letters = set(code)

    axis_letter_counts = [
        len(letters & axis_pair)
        for axis_pair in _AXIS_PAIRS
    ]

    return sorted(axis_letter_counts) == [0, 1, 1]
