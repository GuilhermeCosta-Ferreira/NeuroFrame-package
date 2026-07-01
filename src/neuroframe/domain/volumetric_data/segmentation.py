# ================================================================
# 0. Section: IMPORTS
# ================================================================
import polars as pl
import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass
from typing_extensions import Self

from .volume_key import VolumeKey
from .volume_data import VolumeData
from ..orientation import Orientation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(kw_only=True)
class Segmentation(VolumeData):
    look_up: pl.DataFrame
    background_segment: int

    @property
    def segment_labels(self) -> NDArray:
        return np.unique(self.data)

    def __init__(
        self,
        *,
        data: NDArray,
        look_up: pl.DataFrame,
        background_segment: int,
        resolution: tuple[float, float, float],
        unit: tuple[str, str, str],
        orientation: Orientation,
    ) -> None:
        self.look_up = look_up
        self.background_segment = background_segment

        super().__init__(
            data=data,
            key=VolumeKey.SEGMENTATION,
            resolution=resolution,
            unit=unit,
            orientation=orientation,
            contains_labels=True,
        )

    def copy_with(
        self,
        *,
        data: NDArray | None = None,
        key: VolumeKey | None = None,
        resolution: tuple[float, float, float] | None = None,
        unit: tuple[str, str, str] | None = None,
        orientation: Orientation | None = None,
        contains_labels: bool | None = None,
        look_up: pl.DataFrame | None = None,
        background_segment: int | None = None,
    ) -> Self:
        if key is not None and key != VolumeKey.SEGMENTATION:
            raise ValueError("Segmentation key must remain VolumeKey.SEGMENTATION")

        if contains_labels is not None and contains_labels is not True:
            raise ValueError("Segmentation contains_labels must remain True")

        return type(self)(
            data=self.data if data is None else data,
            look_up=self.look_up if look_up is None else look_up,
            background_segment=(
                self.background_segment
                if background_segment is None
                else background_segment
            ),
            resolution=self.resolution if resolution is None else resolution,
            unit=self.unit if unit is None else unit,
            orientation=self.orientation if orientation is None else orientation,
        )
