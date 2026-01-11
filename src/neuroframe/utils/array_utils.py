# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Quantification
# ================================================================
def count_voxels(input: np.ndarray) -> int:
    nr_voxels = np.sum(np.where(input > 0, 1, 0))
    
    return nr_voxels



# ================================================================
# 2. Section: Shapes
# ================================================================
def enlarge_shape(array: np.ndarray, reference_array: np.ndarray) -> np.ndarray:
    for axis in range(array.ndim):
        if array.shape[axis] >= reference_array.shape[axis]:
            return array

        diff = reference_array.shape[axis] - array.shape[axis]
        pad_before = diff // 2
        pad_after = diff - pad_before

        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (pad_before, pad_after)
        array = np.pad(array, pad_width, mode="constant")

    return array