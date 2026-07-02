# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing_extensions import Any, Self
from dataclasses import dataclass, replace

from .plot_key import PlotKey



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(kw_only=True)
class PlotSpec:
    key: PlotKey
    title: str
    x_label: str
    y_label: str
    parameters: dict[str, Any]



    # ================================================================
    # 0. Section: Copy with
    # ================================================================
    def copy_with(
        self,
        *,
        key: PlotKey | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Self:
        return replace(
            self,
            key=self.key if key is None else key,
            title=self.title if title is None else title,
            x_label=self.x_label if x_label is None else x_label,
            y_label=self.y_label if y_label is None else y_label,
            parameters=self.parameters if parameters is None else parameters,
        )
