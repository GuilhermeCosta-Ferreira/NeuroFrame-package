# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

import numpy as np
import polars as pl

from numpy.typing import NDArray

from neuroframe.domain.mouse_data import MouseData
from neuroframe.domain.orientation import Orientation
from neuroframe.domain.orientation_2d import Orientation2D
from neuroframe.domain.planar_data import PlaneData, PlaneKey, SkullProjection
from neuroframe.domain.volumetric_data import MRI, Segmentation, VolumeData, VolumeKey
from neuroframe.domain.secondary_data import (
    PlotKey,
    PlotSpec,
    SeriesData,
    SeriesKey,
    TableData,
    TableKey,
)



# ================================================================
# 2. Section: Fixtures
# ================================================================
@pytest.fixture
def mri() -> MRI:
    return MRI(
        data=np.zeros((4, 5, 6), dtype=np.float32),
        resolution=(1.0, 1.0, 1.0),
        unit=("mm", "mm", "mm"),
        orientation=Orientation("psl"),
    )

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

@pytest.fixture
def volumes(mri: MRI, segmentation: Segmentation) -> dict[VolumeKey, VolumeData]:
    return {
        VolumeKey.MRI: mri,
        VolumeKey.SEGMENTATION: segmentation,
    }

@pytest.fixture
def skull_projection() -> SkullProjection:
    return SkullProjection(
        data=np.zeros((4, 5), dtype=np.float32),
        resolution=(25.0, 25.0),
        unit=("um", "um"),
        orientation=Orientation2D("ra"),
        method_name="mean"
    )

@pytest.fixture
def planes(skull_projection: SkullProjection) -> dict[PlaneKey, PlaneData]:
    return {
        PlaneKey.SKULL_PROJECTION: skull_projection
    }

@pytest.fixture
def mouse_data(volumes: dict[VolumeKey, VolumeData], planes: dict[PlaneKey, PlaneData]) -> MouseData:
    return MouseData(
        volumes=volumes,
        planes=planes,
        series={},
        tables={},
        plot_specs={},
    )



# ================================================================
# 3. Section: Construction Tests
# ================================================================
@pytest.mark.unit
def test_mouse_data_stores_expected_fields(
    mouse_data: MouseData,
    volumes: dict[VolumeKey, VolumeData],
    planes: dict[PlaneKey, PlaneData],
) -> None:
    assert mouse_data.volumes is volumes
    assert mouse_data.planes is planes
    assert mouse_data.series == {}
    assert mouse_data.tables == {}
    assert mouse_data.plot_specs == {}

@pytest.mark.unit
def test_mouse_data_accepts_empty_dictionaries() -> None:
    mouse = MouseData(
        volumes={},
        planes={},
        series={},
        tables={},
        plot_specs={},
    )

    assert mouse.volumes == {}
    assert mouse.planes == {}
    assert mouse.series == {}
    assert mouse.tables == {}
    assert mouse.plot_specs == {}



# ================================================================
# 4. Section: copy_with Tests
# ================================================================
# ================================================================
# 5. Section: Parametrized copy_with Tests
# ================================================================
@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name, replacement",
    [
        ("volumes", {}),
        ("planes", {}),
        ("series", {}),
        ("tables", {}),
        ("plot_specs", {}),
    ],
    ids=["volumes", "planes", "series", "tables", "plot_specs"],
)
def test_copy_with_can_replace_each_field(
    mouse_data: MouseData,
    field_name: str,
    replacement: dict,
) -> None:
    copied = mouse_data.copy_with(**{field_name: replacement})

    assert getattr(copied, field_name) is replacement

    for other_field_name in (
        "volumes",
        "planes",
        "series",
        "tables",
        "plot_specs",
    ):
        if other_field_name == field_name:
            continue

        assert getattr(copied, other_field_name) == getattr(
            mouse_data,
            other_field_name,
        )

@pytest.mark.unit
def test_copy_with_can_replace_all_fields(
    mouse_data: MouseData,
) -> None:
    new_volumes: dict[VolumeKey, VolumeData] = {}
    new_planes: dict[PlaneKey, PlaneData] = {}
    new_series: dict[SeriesKey, SeriesData] = {}
    new_tables: dict[TableKey, TableData] = {}
    new_plot_specs: dict[PlotKey, PlotSpec] = {}

    copied = mouse_data.copy_with(
        volumes=new_volumes,
        planes=new_planes,
        series=new_series,
        tables=new_tables,
        plot_specs=new_plot_specs,
    )

    assert copied.volumes is new_volumes
    assert copied.planes is new_planes
    assert copied.series is new_series
    assert copied.tables is new_tables
    assert copied.plot_specs is new_plot_specs
