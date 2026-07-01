# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from itertools import permutations

from neuroframe.domain.orientation import Orientation

_AXIS_PAIRS = (
    frozenset(("p", "a")),
    frozenset(("i", "s")),
    frozenset(("r", "l")),
)



# ================================================================
# 1. Section: Contract Tests
# ================================================================
@pytest.mark.unit
def test_orientation_contains_48_members() -> None:
    assert len(Orientation) == 48

@pytest.mark.unit
def test_orientation_contains_all_and_only_valid_values() -> None:
    expected_values = _expected_orientation_values()
    actual_values = {member.value for member in Orientation}

    assert actual_values == expected_values

@pytest.mark.unit
def test_orientation_has_no_duplicate_values() -> None:
    values = [member.value for member in Orientation]

    assert len(values) == len(set(values))

@pytest.mark.unit
@pytest.mark.parametrize("orientation", list(Orientation), ids=lambda item: item.name)
def test_orientation_names_are_uppercase_and_values_are_lowercase(
    orientation: Orientation,
) -> None:
    assert orientation.name == orientation.name.upper()
    assert orientation.value == orientation.value.lower()
    assert orientation.name == orientation.value.upper()

@pytest.mark.unit
@pytest.mark.parametrize("orientation", list(Orientation), ids=lambda item: item.name)
def test_orientation_values_have_three_unique_letters(
    orientation: Orientation,
) -> None:
    value = orientation.value

    assert len(value) == 3
    assert len(set(value)) == 3

@pytest.mark.unit
@pytest.mark.parametrize("orientation", list(Orientation), ids=lambda item: item.name)
def test_orientation_values_contain_exactly_one_letter_from_each_axis(
    orientation: Orientation,
) -> None:
    assert _contains_exactly_one_letter_from_each_axis(orientation.value)

@pytest.mark.unit
def test_orientation_order_is_deterministic_and_sorted_by_name() -> None:
    names = [member.name for member in Orientation]

    assert names == sorted(names)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _expected_orientation_values() -> set[str]:
    expected: set[str] = set()

    for pa in ("p", "a"):
        for is_ in ("i", "s"):
            for rl in ("r", "l"):
                selected_letters = (pa, is_, rl)

                for ordered_letters in permutations(selected_letters):
                    expected.add("".join(ordered_letters))

    return expected


def _contains_exactly_one_letter_from_each_axis(code: str) -> bool:
    letters = set(code)

    return all(len(letters & axis_pair) == 1 for axis_pair in _AXIS_PAIRS)
