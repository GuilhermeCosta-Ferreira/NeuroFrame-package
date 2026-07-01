# ================================================================
# 0. Section: IMPORTS
# ================================================================
from enum import StrEnum
from itertools import product, permutations



# ================================================================
# 1. Section: Constants and Helpers
# ================================================================
_ORIENTATION_GROUPS = (
    ("p", "a"),
    ("i", "s"),
    ("r", "l"),
)

def _generate_orientation_values() -> dict[str, str]:
    values: dict[str, str] = {}

    for selected_letters in product(*_ORIENTATION_GROUPS):
        for ordered_letters in permutations(selected_letters):
            code = "".join(ordered_letters)
            values[code.upper()] = code

    return dict(sorted(values.items()))



# ================================================================
# 3. Section: Enums
# ================================================================
Orientation = StrEnum(
    "Orientation",
    _generate_orientation_values(),
)
