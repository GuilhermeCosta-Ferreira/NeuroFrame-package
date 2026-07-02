# ================================================================
# 0. Section: IMPORTS
# ================================================================
from enum import Enum
from itertools import product, permutations, combinations



# ================================================================
# 1. Section: Constants and Helpers
# ================================================================
class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


_ORIENTATION_GROUPS = (
    ("p", "a"),
    ("i", "s"),
    ("r", "l"),
)


def _generate_orientation_2d_values() -> dict[str, str]:
    values: dict[str, str] = {}

    for selected_groups in combinations(_ORIENTATION_GROUPS, 2):
        for selected_letters in product(*selected_groups):
            for ordered_letters in permutations(selected_letters):
                code = "".join(ordered_letters)
                values[code.upper()] = code

    return dict(sorted(values.items()))



# ================================================================
# 3. Section: Enums
# ================================================================
Orientation2D = StrEnum(
    "Orientation2D",
    _generate_orientation_2d_values(),
)
