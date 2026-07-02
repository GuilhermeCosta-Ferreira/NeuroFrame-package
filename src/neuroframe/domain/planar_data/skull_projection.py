# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from dataclasses import dataclass
from typing_extensions import Self

from .plane_key import PlaneKey
from .plane_data import PlaneData
from ..orientation_2d import Orientation2D



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(kw_only=True)
class SkullProjection(PlaneData):
    method_name: str

    def __init__(
        self,
        *,
        data: NDArray,
        resolution: tuple[float, float],
        unit: tuple[str, str],
        orientation: Orientation2D,
        method_name: str,
    ) -> None:
        self.method_name = method_name

        super().__init__(
            data=data,
            key=PlaneKey.SKULL_PROJECTION,
            resolution=resolution,
            unit=unit,
            orientation=orientation,
            contains_labels=False,
        )



    # ================================================================
    # 0. Section: Copy With
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
        method_name: str | None = None,
    ) -> Self:
        if key is not None and key != PlaneKey.SKULL_PROJECTION:
            raise ValueError("Skull projection key must remain PlaneKey.SKULL_PROJECTION")

        if contains_labels is not None and contains_labels is not False:
            raise ValueError("Segmentation contains_labels must remain False")

        return type(self)(
            data=self.data if data is None else data,
            resolution=self.resolution if resolution is None else resolution,
            unit=self.unit if unit is None else unit,
            orientation=self.orientation if orientation is None else orientation,
            method_name=self.method_name if method_name is None else method_name,
        )
