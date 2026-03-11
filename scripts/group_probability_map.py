# ================================================================
# 0. Section: IMPORTS
# ================================================================
import re

import numpy as np
import pandas as pd

from pathlib import Path
from neuroframe import Mouse

from neuroframe.group_analysis import (
    get_average_volume,
    pick_closest_to_average
)
from neuroframe.pipeline import layer_colapsing


# ================================================================
# 1. Section: INPUTS
# ================================================================
BASE_DIR: Path = Path(__file__).resolve().parents[2]
MICE_FOLDER: Path = BASE_DIR / "data"
REFERENCE_MOUSE: Path | None = MICE_FOLDER / "S872"
SEGMENT_INFO_PATH: Path = BASE_DIR / "data" / "annotations_info.csv"



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def get_mouse_folders(folder: Path) -> list[Path]:
    pattern = re.compile(r"^[A-Z]\d{3}$")
    return sorted([p for p in folder.iterdir() if p.is_dir() and pattern.match(p.name)])

def get_unique_segments(mice: list[Path]) -> np.ndarray:
    uniques = []
    for mouse in mice:
        mouse = Mouse.from_folder(mouse.name, str(mouse), was_analysed=True)
        labels = mouse.segmentation.labels

        uniques.append(labels)

    return np.unique(np.concatenate(uniques))

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
    reference_mouse = Mouse.from_folder(
        id=REFERENCE_MOUSE.name,
        folder_path=str(REFERENCE_MOUSE),
        was_analysed=True
    )

    # 3. Instantiates the average mouse
    probability_map = np.zeros(reference_mouse.data_shape)
    lateralization_map = np.zeros(reference_mouse.data_shape)
    segmentation_map = np.zeros(reference_mouse.data_shape)
    bl_map = reference_mouse.field_bl.data.copy()

    # 4. Gets all unique segments alonga all mice
    all_segments = get_unique_segments(mice)

    # 5. Gets all BL Translation vectors
    reference_zero = np.argwhere(np.all(bl_map == [0, 0, 0], axis=-1))[0]
    for mouse in mice:
        if(mouse.name == REFERENCE_MOUSE.name):
            continue

        mouse = Mouse.from_folder(mouse.name, str(mouse), True)
        bl_space = mouse.field_bl.data
        hemispheres = mouse.hemisphere.data
        mouse_nedt = mouse.segmentation_nedt
        mouse_segments = mouse.segmentation.data

        mouse_zero = np.argwhere(np.all(bl_space == [0, 0, 0], axis=-1))[0]

        translation = reference_zero - mouse_zero

        import numpy as np

        shifted = np.roll(mouse_segments, shift=tuple(translation), axis=(0, 1, 2))
        break


        # 5.1

    # 6. Loops over every segment
