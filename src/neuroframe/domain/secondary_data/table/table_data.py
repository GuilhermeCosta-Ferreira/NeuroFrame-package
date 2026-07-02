# ================================================================
# 0. Section: IMPORTS
# ================================================================
import polars as pl

from typing_extensions import Self
from dataclasses import dataclass, replace

from .table_key import TableKey



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(kw_only=True)
class TableData:
    data: pl.DataFrame
    key: TableKey
    description: str | None



    # ================================================================
    # 0. Section: Copy with
    # ================================================================
    def copy_with(
        self,
        *,
        data: pl.DataFrame | None = None,
        key: TableKey | None = None,
        description: str | None = None,
    ) -> Self:
        return replace(
            self,
            data=self.data if data is None else data,
            key=self.key if key is None else key,
            description=self.description if description is None else description,
        )
