# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pytest

from neuroframe.domain.volumetric_data.volume_data import VolumeData
from neuroframe.domain.volumetric_data.volume_key import VolumeKey
from neuroframe.domain.orientation import Orientation



# ================================================================
# 1. Section: Fixtures
# ================================================================
@pytest.fixture
def volume_array() -> np.ndarray:
    return np.zeros((4, 5, 6), dtype=np.float32)


@pytest.fixture
def volume_data(volume_array: np.ndarray) -> VolumeData:
    return VolumeData(
        data=volume_array,
        key=VolumeKey.MRI,
        resolution=(25.0, 25.0, 50.0),
        unit=("um", "um", "um"),
        orientation=Orientation("ras"),
    )



# ================================================================
# 2. Section: Construction Tests
# ================================================================
@pytest.mark.unit
def test_volume_data_stores_expected_fields(
    volume_array: np.ndarray, volume_data: VolumeData
) -> None:
    volume = volume_data.copy_with(contains_labels=True)

    assert volume.data is volume_array
    assert volume.key is VolumeKey.MRI
    assert volume.resolution == (25.0, 25.0, 50.0)
    assert volume.unit == ("um", "um", "um")
    assert volume.orientation is Orientation("ras")
    assert volume.contains_labels is True

@pytest.mark.unit
def test_volume_data_contains_labels_defaults_to_false(
    volume_data: VolumeData,
) -> None:
    assert volume_data.contains_labels is False

@pytest.mark.unit
def test_volume_data_shape_returns_data_shape(volume_data: VolumeData) -> None:
    assert volume_data.shape == (4, 5, 6)

@pytest.mark.unit
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (4, 5),
        (4, 5, 6, 7),
    ],
    ids=["1d", "2d", "4d"],
)
def test_volume_data_rejects_non_3d_arrays(shape: tuple[int, ...]) -> None:
    data = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="3D array"):
        VolumeData(
            data=data,
            key=VolumeKey.MRI,
            resolution=(25.0, 25.0, 50.0),
            unit=("um", "um", "um"),
            orientation=Orientation("ras"),
        )
