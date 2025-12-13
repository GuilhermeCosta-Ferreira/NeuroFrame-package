# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
import numpy as np

from ..logger import logger
from .registrator_utils import convert_input
from .Definers import Definers



class RegistratorSupport(Definers):
    # ================================================================
    # 1. Section: Resample
    # ================================================================
    def resample(self, fixed_image: sitk.Image | np.ndarray, moving_image: sitk.Image | np.ndarray, transform: sitk.Transform) -> sitk.Image:
        
        fixed_image = convert_input(fixed_image)
        moving_image = convert_input(moving_image)

        # Create a resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler = self.define_resample_interpolator(resampler)
        resampler.SetTransform(transform)
        logger.debug("Resampler setup complete.")

        # Resample the image
        resampled_image = resampler.Execute(moving_image)
        logger.debug("Resampling executed.")

        return resampled_image
    


    # ================================================================
    # 2. Section: Apply Transform
    # ================================================================
    def apply_transform(self, image: np.ndarray | sitk.Image, transform: sitk.Transform) -> np.ndarray:
        
        image = convert_input(image)

        resampled = sitk.Resample(
            image1           = image,
            transform        = transform)
        logger.debug("Applied transform to image.")

        transformed_image = sitk.GetArrayFromImage(resampled)

        return transformed_image