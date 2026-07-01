# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from neuroframe.domain.volumetric_data.volume_key import VolumeKey



# ================================================================
# 1. Section: Functions
# ================================================================
@pytest.mark.unit
def test_volume_key_can_be_created_from_value() -> None:
    assert VolumeKey("default") is VolumeKey.DEFAULT
    assert VolumeKey("mri") is VolumeKey.MRI
    assert VolumeKey("uct") is VolumeKey.CT
    assert VolumeKey("segmentation") is VolumeKey.SEGMENTATION
