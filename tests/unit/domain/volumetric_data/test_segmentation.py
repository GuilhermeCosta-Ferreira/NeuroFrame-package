# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import polars as pl
import pytest

from numpy.typing import NDArray
from polars.testing import assert_frame_equal

from neuroframe.domain.orientation import Orientation
from neuroframe.domain.volumetric_data.segmentation import Segmentation
from neuroframe.domain.volumetric_data.volume_key import VolumeKey



# ================================================================
# 1. Section: Fixtures
# ================================================================
@pytest.fixture
def segmentation_array() -> NDArray:
    return np.array(
        [
            [[0, 1, 1], [2, 2, 0]],
            [[3, 3, 1], [0, 2, 3]],
        ],
        dtype=np.uint16,
    )

@pytest.fixture
def segmentation_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "segment_id": [0, 1, 2, 3],
            "name": ["background", "region_a", "region_b", "region_c"],
        }
    )

@pytest.fixture
def segmentation(
    segmentation_array: NDArray,
    segmentation_lookup: pl.DataFrame,
) -> Segmentation:
    return Segmentation(
        data=segmentation_array,
        look_up=segmentation_lookup,
        background_segment=0,
        resolution=(25.0, 25.0, 50.0),
        unit=("um", "um", "um"),
        orientation=Orientation("ras"),
    )



# ================================================================
# 2. Section: Construction Tests
# ================================================================
@pytest.mark.unit
def test_segmentation_stores_expected_fields(
    segmentation_array: NDArray,
    segmentation_lookup: pl.DataFrame,
    segmentation: Segmentation,
) -> None:
    assert segmentation.data is segmentation_array
    assert_frame_equal(segmentation.look_up, segmentation_lookup)
    assert segmentation.background_segment == 0
    assert segmentation.resolution == (25.0, 25.0, 50.0)
    assert segmentation.unit == ("um", "um", "um")
    assert segmentation.orientation is Orientation("ras")

@pytest.mark.unit
def test_segmentation_always_has_segmentation_key(
    segmentation: Segmentation,
) -> None:
    assert segmentation.key is VolumeKey.SEGMENTATION

@pytest.mark.unit
def test_segmentation_always_contains_labels(
    segmentation: Segmentation,
) -> None:
    assert segmentation.contains_labels is True

@pytest.mark.unit
def test_segmentation_shape_returns_data_shape(
    segmentation: Segmentation,
) -> None:
    assert segmentation.shape == (2, 2, 3)
