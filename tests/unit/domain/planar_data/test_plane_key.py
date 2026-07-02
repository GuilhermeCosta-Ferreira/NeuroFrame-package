# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from neuroframe.domain.planar_data.plane_key import PlaneKey



# ================================================================
# 1. Section: Functions
# ================================================================
@pytest.mark.unit
def test_plane_key_can_be_created_from_value() -> None:
    assert PlaneKey("default") is PlaneKey.DEFAULT
    assert PlaneKey("skull_projection") is PlaneKey.SKULL_PROJECTION
