# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from typing_extensions import Self
from dataclasses import dataclass, replace

from .series_key import SeriesKey



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(kw_only=True)
class SeriesData:
    x: NDArray
    y: NDArray
    key: SeriesKey
    x_label: str
    y_label: str
    x_unit: str | None
    y_unit: str | None



    # ================================================================
    # 0. Section: Copy with
    # ================================================================
    def copy_with(
        self,
        *,
        x: NDArray | None = None,
        y: NDArray | None = None,
        key: SeriesKey | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        x_unit: str | None = None,
        y_unit: str | None = None,
    ) -> Self:
        return replace(
            self,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            key=self.key if key is None else key,
            x_label=self.x_label if x_label is None else x_label,
            y_label=self.y_label if y_label is None else y_label,
            x_unit=self.x_unit if x_unit is None else x_unit,
            y_unit=self.y_unit if y_unit is None else y_unit,
        )
