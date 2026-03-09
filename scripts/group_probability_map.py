# ================================================================
# 0. Section: IMPORTS
# ================================================================
import re

from pathlib import Path
from neuroframe import Mouse

from neuroframe.group_analysis import (
    get_average_volume,
    pick_closest_to_average
)


# ================================================================
# 1. Section: INPUTS
# ================================================================
BASE_DIR: Path = Path(__file__).resolve().parents[2]
MICE_FOLDER: Path = BASE_DIR / "data"
REFERENCE_MOUSE: Path | None = MICE_FOLDER / "S872"



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def get_mouse_folders(folder: Path) -> list[Path]:
    pattern = re.compile(r"^[A-Z]\d{3}$")
    return sorted([p for p in folder.iterdir() if p.is_dir() and pattern.match(p.name)])


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Get all the mice in the study
    mice = get_mouse_folders(MICE_FOLDER)

    # 2. Get the reference mouse if needed, once is done, just update INPUT
    if(REFERENCE_MOUSE is None):
        average_volume = get_average_volume(mice)
        REFERENCE_MOUSE = pick_closest_to_average(mice, average_volume)

    # 3.
