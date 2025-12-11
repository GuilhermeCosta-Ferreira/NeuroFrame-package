# ================================================================
# 0. Section: Imports
# ================================================================
from ..mouse_data import MicroCT, MRI, Segmentation
from ._dunders import Dunders
from ._properties import Properties
from ._plots import Plots



# ================================================================
# 1. Section: Mouse Classes
# ================================================================
class Mouse(Dunders, Properties, Plots):
    def __init__(self, id: str, mri_path: str, ct_path: str, segmentations_path: str) -> None:
        self.micro_ct = MicroCT(ct_path)
        self.mri = MRI(mri_path)
        self.segmentations = Segmentation(segmentations_path)

        self.paths = {
            'ct_path': ct_path,
            'mri_path': mri_path,
            'segmentations_path': segmentations_path
        }

        self.id = id