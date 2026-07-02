# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from typing_extensions import Self
from dataclasses import dataclass, replace

from .plane_key import PlaneKey
from ..orientation_2d import Orientation2D



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(kw_only=True)
class PlaneData:
    data : NDArray
    key: PlaneKey
    resolution: tuple[float, float]
    unit: tuple[str, str]
    orientation: Orientation2D
    contains_labels: bool

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape)  # type: ignore



    # ================================================================
    # 2. Section: Dunders
    # ================================================================
    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError("data must be a 2D array")



    # ================================================================
    # 3. Section: Methods
    # ================================================================
    def copy_with(
        self,
        *,
        data: NDArray | None = None,
        key: PlaneKey | None = None,
        resolution: tuple[float, float] | None = None,
        unit: tuple[str, str] | None = None,
        orientation: Orientation2D | None = None,
        contains_labels: bool | None = None,
    ) -> Self:
        return replace(
            self,
            data=self.data if data is None else data,
            key=self.key if key is None else key,
            resolution=self.resolution if resolution is None else resolution,
            unit=self.unit if unit is None else unit,
            orientation=self.orientation if orientation is None else orientation,
            contains_labels=self.contains_labels if contains_labels is None else contains_labels,
        )
