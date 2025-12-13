# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from ..logger import logger



# ================================================================
# 1. Section: Shapes
# ================================================================
def assert_same_shape(array1: np.ndarray, array2: np.ndarray) -> None:
    if array1.shape != array2.shape:
        logger.warning(f"Array shapes are not the same: {array1.shape} vs {array2.shape}")