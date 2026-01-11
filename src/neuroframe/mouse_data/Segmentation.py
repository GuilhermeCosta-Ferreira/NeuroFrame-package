# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from .MedicalImage import MedicalImage
from ..utils import normalize

from .dunders._Segmentation import Dunders
from .properties._Segmentation import Properties



# ================================================================
# 1. Section: MRI Class
# ================================================================
class Segmentation(Dunders, Properties, MedicalImage):
    def __init__(self, path):
        super().__init__(path)