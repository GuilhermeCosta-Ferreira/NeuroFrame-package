# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd
import numpy as np

from pprint import pprint
from pathlib import Path
from neuroframe import Mouse
from neuroframe.mouse_data import (
    Hemisphere,
    SegmentationEDT,
    SegmentationNEDT,
    FieldBL,
    MRI,
    MicroCT,
    Segmentation
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("../data")
MOUSE_FOLDER: Path = Path("../data/P874")
MOUSE_ID: str = MOUSE_FOLDER.stem
WT_MOUSE_FOLDER: Path = Path("../data/W001")
WT_MOUSE_ID: Path = WT_MOUSE_FOLDER.stem

SEGMENT_INFO_PATH: Path = Path("data/annotations_info.csv")


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def get_pattern_file(mouse_folder: Path, pattern: str = "*_sides.nii.gz*") -> Path:
    # 1. Get the matches
    it = mouse_folder.rglob(pattern)
    matches = sorted([p for p in it if p.is_file()])

    if len(matches) > 1:
        print(
            f"Found multiple '{pattern}' files in {mouse_folder}. "
            f"Using the first: {matches[0].name}. All: {[m.name for m in matches]}"
        )

    return matches[0]


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    mice = [p for p in DATA_FOLDER.iterdir() if p.is_dir() and p.name != "W001"]
    data = {}
    for mouse_info in mice:
        mouse = Mouse.from_folder(mouse_info.stem, str(mouse_info))
        segmentation_info = pd.read_csv(SEGMENT_INFO_PATH)

        mri_path = get_pattern_file(mouse_info, "*_proc_mri.nii.gz*")
        ct_path = get_pattern_file(mouse_info, "*_proc_ct.nii.gz*")
        seg_path = get_pattern_file(mouse_info, "*_proc_seg.nii.gz*")
        hemisphere_path = get_pattern_file(mouse_info, "*_sides.nii.gz*")
        edt_path = get_pattern_file(mouse_info, "*_edt.nii.gz*")
        nedt_path = get_pattern_file(mouse_info, "*_nedt.nii.gz*")
        bl_space_path = get_pattern_file(mouse_info, "*_bl_space.nii.gz*")

        mouse.mri = MRI(str(mri_path))
        mouse.micro_ct = MicroCT(str(ct_path))
        mouse.segmentation = Segmentation(str(seg_path))
        mouse.add_path(hemisphere_path, Hemisphere)
        mouse.add_path(edt_path, SegmentationEDT)
        mouse.add_path(nedt_path, SegmentationNEDT)
        mouse.add_path(bl_space_path, FieldBL)

        stn_id = 470
        segment = np.where(mouse.segmentation.data == stn_id, mouse.hemisphere.data, 0)
        space = mouse.field_bl.data

        points = {}
        for label, name in ((1, "left"), (2, "right")):
            coords = np.argwhere(segment == label)   # rows are [dim1, dim2, dim3]

            if coords.size == 0:
                points[name] = None
                continue

            idx = np.argmin(coords[:, 1])           # column 1 = dim2
            dim1, dim2, dim3 = tuple(coords[idx])
            sp1, sp2, sp3 = np.round(space[dim1, dim2, dim3], 2)
            points[name] = np.array([sp3, sp2, sp1])

        print(points)

        data[mouse.id] = points

    pprint(data)

    df = pd.DataFrame(data)
    df.to_csv("stn_urgent.csv")

    row_mean = df.apply(lambda row: np.round(np.mean(np.vstack(row), axis=0), 2), axis=1)

    row_mean.to_csv("stn_urgent_mean.csv")

    print(row_mean)
