# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

import numpy as np

from numpy.typing import NDArray

from neuroframe.domain.orientation_2d import Orientation2D
from neuroframe.domain.planar_data.plane_key import PlaneKey
from neuroframe.domain.planar_data.plane_data import PlaneData



# ================================================================
# 1. Section: Fixtures
# ================================================================
@pytest.fixture
def plane_array() -> NDArray:
    return np.zeros((4, 5), dtype=np.float32)


@pytest.fixture
def plane_data(plane_array: NDArray) -> PlaneData:
    return PlaneData(
        data=plane_array,
        key=PlaneKey.SKULL_PROJECTION,
        resolution=(25.0, 25.0),
        unit=("um", "um"),
        orientation=Orientation2D("ra"),
        contains_labels=False
    )



# ================================================================
# 2. Section: Construction Tests
# ================================================================
@pytest.mark.unit
def test_plane_data_stores_expected_fields(
    plane_array: NDArray, plane_data: PlaneData
) -> None:
    plane = plane_data.copy_with(contains_labels=True)

    assert plane.data is plane_array
    assert plane.key is PlaneKey.SKULL_PROJECTION
    assert plane.resolution == (25.0, 25.0)
    assert plane.unit == ("um", "um")
    assert plane.orientation is Orientation2D("ra")
    assert plane.contains_labels is True

@pytest.mark.unit
def test_plane_data_contains_labels_defaults_to_false(
    plane_data: PlaneData,
) -> None:
    assert plane_data.contains_labels is False

@pytest.mark.unit
def test_plane_data_shape_returns_data_shape(plane_data: PlaneData) -> None:
    assert plane_data.shape == (4, 5)

@pytest.mark.unit
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (4, 5, 6),
        (4, 5, 6, 7),
    ],
    ids=["1d", "3d", "4d"],
)
def test_plane_data_rejects_non_3d_arrays(shape: tuple[int, ...]) -> None:
    data = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="2D array"):
        PlaneData(
            data=data,
            key=PlaneKey.SKULL_PROJECTION,
            resolution=(25.0, 25.0),
            unit=("um", "um"),
            orientation=Orientation2D("ra"),
            contains_labels=False
        )
