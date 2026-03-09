# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path
from ..mouse import Mouse


# ================================================================
# 1. Section: Pick the mouse with the brain closest to average
# ================================================================
def pick_closest_to_average(mice: list[Path], average_volume: float) -> Path:
    diffs = []
    for mouse in mice:
        # 1. Extract the MRI data
        mouse_class = Mouse.from_folder(id=mouse.name, folder_path=str(mouse))
        mri = mouse_class.mri.nib
        mri_arr = mouse_class.mri.data

        # 2. Get ingredients for volume calculation
        mri_mask = np.where(mri_arr > 0, 1, 0)
        voxel_size = np.round(mri.header.get_zooms(), 3)
        voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]

        # 3. Calculate brain volume
        brain_volume = np.sum(mri_mask) * voxel_volume
        diffs.append(abs(brain_volume - average_volume))

        # 4. Close the Nifti file
        mri, mri_arr, mri_mask = None, None, None

    closest_index = np.argmin(diffs)
    closest_mouse_path = mice[closest_index]

    return closest_mouse_path


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Get the average volumes
# ──────────────────────────────────────────────────────
def get_average_volume(mice: list[Path]) -> float:
    volumes = []
    for mouse in mice:
        # Extract the MRI data
        mouse_class = Mouse.from_folder(id=mouse.name, folder_path=str(mouse))
        mri = mouse_class.mri.nib
        mri_arr = mouse_class.mri.data

        # Get ingredients for volume calculation
        mri_mask = np.where(mri_arr > 0, 1, 0)
        voxel_size = np.round(mri.header.get_zooms(), 3)
        voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]

        # Calculate brain volume
        brain_volume = np.sum(mri_mask) * voxel_volume
        volumes.append(brain_volume)

        # Close the Nifti file
        mri, mri_arr, mri_mask = None, None, None

    # Calculate average volume
    average_volume = np.mean(volumes)

    return average_volume
