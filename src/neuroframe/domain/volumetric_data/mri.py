# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from dataclasses import dataclass

from .volume_key import VolumeKey
from .volume_data import VolumeData
from ..orientation import Orientation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MRI(VolumeData):
    def __init__(
        self,
        *,
        data: NDArray,
        resolution: tuple[float, float, float],
        unit: tuple[str, str, str],
        orientation: Orientation,
    ) -> None:
        super().__init__(
            data=data,
            key=VolumeKey.MRI,
            resolution=resolution,
            unit=unit,
            orientation=orientation,
            contains_labels=False,
        )
