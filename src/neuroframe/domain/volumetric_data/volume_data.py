# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from typing_extensions import Self
from dataclasses import dataclass, replace

from .volume_key import VolumeKey
from ..orientation import Orientation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class VolumeData:
    data : NDArray
    key: VolumeKey
    resolution: tuple[float, float, float]
    unit: tuple[str, str, str]
    orientation: Orientation

    contains_labels: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore



    # ================================================================
    # 2. Section: Dunders
    # ================================================================
    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError("data must be a 3D array")



    # ================================================================
    # 3. Section: Methods
    # ================================================================
    def copy_with(
        self,
        *,
        data: NDArray | None = None,
        key: VolumeKey | None = None,
        resolution: tuple[float, float, float] | None = None,
        unit: tuple[str, str, str] | None = None,
        orientation: Orientation | None = None,
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
