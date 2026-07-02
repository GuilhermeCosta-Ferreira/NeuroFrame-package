# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import polars as pl
import pytest

from numpy.typing import NDArray
from polars.testing import assert_frame_equal

from neuroframe.domain.orientation_2d import Orientation2D
from neuroframe.domain.planar_data.skull_projection import SkullProjection
from neuroframe.domain.planar_data.plane_key import PlaneKey



# ================================================================
# 1. Section: Fixtures
# ================================================================
@pytest.fixture
def skull_array() -> NDArray:
    return np.zeros((4, 5), dtype=np.float32)

@pytest.fixture
def skull_projection(
    skull_array: NDArray,
) -> SkullProjection:
    return SkullProjection(
        data=skull_array,
        resolution=(25.0, 25.0),
        unit=("um", "um"),
        orientation=Orientation2D("ra"),
        method_name="mean"
    )



# ================================================================
# 2. Section: Construction Tests
# ================================================================
@pytest.mark.unit
def test_skull_projection_stores_expected_fields(
    skull_array: NDArray,
    skull_projection: SkullProjection,
) -> None:
    assert skull_projection.data is skull_array
    assert skull_projection.resolution == (25.0, 25.0)
    assert skull_projection.unit == ("um", "um")
    assert skull_projection.orientation is Orientation2D("ra")

@pytest.mark.unit
def test_skull_projectio_always_has_skull_key(
    skull_projection: SkullProjection,
) -> None:
    assert skull_projection.key is PlaneKey.SKULL_PROJECTION

@pytest.mark.unit
def test_skull_always_not_contains_labels(
    skull_projection: SkullProjection,
) -> None:
    assert skull_projection.contains_labels is False

@pytest.mark.unit
def test_skull_shape_returns_data_shape(
    skull_projection: SkullProjection,
) -> None:
    assert skull_projection.shape == (4, 5)
