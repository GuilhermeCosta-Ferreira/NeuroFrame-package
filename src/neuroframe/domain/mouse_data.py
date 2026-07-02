# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing_extensions import Self
from dataclasses import dataclass, replace

from .volumetric_data import VolumeData, VolumeKey
from .planar_data import PlaneData, PlaneKey
from .secondary_data import (
    PlotKey, PlotSpec,
    SeriesData, SeriesKey,
    TableData, TableKey,
)



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MouseData:
    volumes: dict[VolumeKey, VolumeData]
    planes: dict[PlaneKey, PlaneData]
    series: dict[SeriesKey, SeriesData]
    tables: dict[TableKey, TableData]
    plot_specs: dict[PlotKey, PlotSpec]



    # ================================================================
    # 0. Section: Copy with
    # ================================================================
    def copy_with(
        self,
        *,
        volumes: dict[VolumeKey, VolumeData] | None = None,
        planes: dict[PlaneKey, PlaneData] | None = None,
        series: dict[SeriesKey, SeriesData] | None = None,
        tables: dict[TableKey, TableData] | None = None,
        plot_specs: dict[PlotKey, PlotSpec] | None = None,
    ) -> Self:
        return replace(
            self,
            volumes=self.volumes.copy() if volumes is None else volumes,
            planes=self.planes.copy() if planes is None else planes,
            series=self.series.copy() if series is None else series,
            tables=self.tables.copy() if tables is None else tables,
            plot_specs=self.plot_specs.copy() if plot_specs is None else plot_specs,
        )
