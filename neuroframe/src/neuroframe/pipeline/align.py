# ================================================================
# 0. Section: Imports
# ================================================================
import os

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from scipy.ndimage import zoom

from ..utils import count_voxels, enlarge_shape
from ..mouse_data import Segmentation
from ..mouse import Mouse
from ..logger import logger


# ──────────────────────────────────────────────────────
# 0.1 Subsection: Universal Constants
# ──────────────────────────────────────────────────────
ALLEN_TEMPLATE = Segmentation("src/neuroframe/templates/allen_brain_25μm_ccf_2017.nii.gz")



# ================================================================
# 1. Section: Align the Mouse to the Allen Template
# ================================================================
def align_to_allen(mouse: Mouse, template: Segmentation = ALLEN_TEMPLATE) -> np.ndarray:
    template_volume = adapt_template(mouse, template)

    # Does the rigid registration

    # Applies the transformation to the mice


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Adapts the Template's Size
# ──────────────────────────────────────────────────────
def adapt_template(mouse: Mouse, template: Segmentation) -> np.ndarray:    
    # Extract the volumes
    mouse_volume = mouse.segmentation.volume
    template_volume = template.volume

    # Compute the volume of the template and mice
    mouse_size = count_voxels(mouse_volume)
    template_size = count_voxels(template_volume)

    # Reduce template volume to match mice volume
    zoom_factor = (mouse_size / template_size) ** (1/3)
    logger.debug(f"Zoom Factor: {zoom_factor.round(2)}")
    template_volume = zoom(template_volume, zoom_factor)
    logger.debug(f"Template shape after zoom: {template_volume.shape}")

    # Fix template shape issue
    template_volume = enlarge_shape(template_volume, mouse_volume)

    logger.debug(f"Template shape after filling in: {template_volume.shape}")
    logger.debug(f"Template is now adapted to the Mouse Volume!")
    return template_volume


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Align Mice to the Template
# ──────────────────────────────────────────────────────
'''
def align_mice(mouse: Mouse, template: np.ndarray, transform: sitk.Transform) -> np.ndarray:
    # Initiates the rigid transformation
    rigid_transform = nc.TransformBrain(res_interpolator="linear")
    rigid_transform_nearest = nc.TransformBrain(res_interpolator="nearest")

    # Apply the rigid transformation to the template and mice volumes
    mri_aligned = rigid_transform.apply_transform(template, mouse.get_mri().get_data(), transform)
    ct_aligned = rigid_transform.apply_transform(template, mouse.get_micro_ct().get_data(), transform)
    seg_aligned = rigid_transform_nearest.apply_transform(template, mouse.get_segmentations().get_data(), transform)

    # Set the aligned data back to the mice object
    mouse.mri.set_data(mri_aligned)
    mouse.micro_ct.set_data(ct_aligned)
    mouse.segmentation.set_data(seg_aligned)

    # Center mice just in case
    #nc.center_mice(mice)

# ================================================================
# 3. Section: Inspect Universal Align
# ================================================================
def inspect_template_orientation(template_volume, mice_volume):
    """
    Inspect the orientation of template and mice volumes by visualizing slice overlays.
    This function creates a figure with 9 subplots arranged in a 3x3 grid to compare the
    template and mice volumes along three orthogonal directions (axial, coronal, and sagittal).
    Both input volumes are first converted into binary representations where any value greater than 0 is set
    to 1 (and non-positive values are set to NaN) to facilitate transparency effects during overlay.

    Parameters:
            template_volume (numpy.ndarray): A 3D numpy array representing the template volume.
            mice_volume (numpy.ndarray): A 3D numpy array representing the mice volume.
    Returns:
            None
    The function does not return any value; it displays the generated plots using matplotlib.
    """
    mice_volume_transparent = np.where(mice_volume > 0, 1, np.nan)
    template_volume_transparent = np.where(template_volume > 0, 1, np.nan)

    # Inspect the template and mice volumes
    plt.figure(figsize=(6, 6))
    plt.subplot(3, 3, 1)
    plt.imshow(template_volume[template_volume.shape[0]//2, :, :], cmap="gray")
    plt.vlines(x=template_volume.shape[2]//2, ymin=0, ymax=template_volume.shape[1]-1, color="red", lw=2)
    plt.title("Template Volume")

    plt.subplot(3, 3, 2)
    plt.imshow(mice_volume[mice_volume.shape[2]//2, :, :], cmap="gray")
    plt.vlines(x=mice_volume.shape[2]//2, ymin=0, ymax=mice_volume.shape[1]-1, color="red", lw=2)
    plt.title("Mice Volume")

    plt.subplot(3, 3, 3)
    plt.imshow(mice_volume_transparent[mice_volume_transparent.shape[0]//2, :, :], cmap="gray")
    plt.imshow(template_volume_transparent[template_volume_transparent.shape[0]//2, :, :], alpha=0.8, cmap="summer")
    plt.vlines(x=template_volume.shape[2]//2, ymin=0, ymax=template_volume.shape[1]-1, color="red", lw=2)
    plt.title("Overlay")

    plt.subplot(3, 3, 4)
    plt.imshow(template_volume[:, template_volume.shape[1]//2, :], cmap="gray")
    plt.vlines(x=template_volume.shape[2]//2, ymin=0, ymax=template_volume.shape[0]-1, color="red", lw=2)
    plt.title("Template Volume")
    plt.subplot(3, 3, 5)
    plt.imshow(mice_volume[:, mice_volume.shape[1]//2, :], cmap="gray")
    plt.vlines(x=template_volume.shape[2]//2, ymin=0, ymax=template_volume.shape[0]-1, color="red", lw=2)
    plt.title("Mice Volume")
    plt.subplot(3, 3, 6)
    plt.imshow(mice_volume_transparent[:, mice_volume_transparent.shape[1]//2, :], cmap="gray")
    plt.imshow(template_volume_transparent[:, template_volume_transparent.shape[1]//2, :], alpha=0.8, cmap="summer")
    plt.vlines(x=template_volume.shape[2]//2, ymin=0, ymax=template_volume.shape[0]-1, color="red", lw=2)
    plt.title("Overlay")
    
    plt.subplot(3, 3, 7)
    plt.imshow(template_volume[:, :, template_volume.shape[2]//2], cmap="gray")
    plt.title("Template Volume")
    plt.subplot(3, 3, 8)
    plt.imshow(mice_volume[:, :, mice_volume.shape[2]//2], cmap="gray")
    plt.title("Mice Volume")
    plt.subplot(3, 3, 9)
    plt.imshow(mice_volume_transparent[:, :, mice_volume_transparent.shape[2]//2], cmap="gray")
    plt.imshow(template_volume_transparent[:, :, template_volume_transparent.shape[2]//2], alpha=0.8, cmap="summer")
    plt.title("Overlay")

    plt.tight_layout()
    plt.show()

def inspect_transformation_slicer(mice: nc.Mice, template: nib.Nifti1Image, new_volume: np.ndarray) -> nib.Nifti1Image:
    """
    Apply transformation on the new_volume using the template's affine information and the properties of mice,
    and save the resulting image for inspection with 3D Slicer.

    Parameters:
        mice: An object that provides access to segmentation data and folder information. It is expected to have
              methods get_segmentations() returning an object with a nibabel image, and get_folder() returning a file path.
        template: An image template object that provides an affine transformation and nibabel image information through get_nib().
                  Its affine attribute and nibabel image properties (such as header zooms) are used to align the new volume.
        new_volume: A numpy array representing the volume data to be transformed and aligned.

    Returns:
        new_image: A nibabel Nifti1Image object containing the transformed volume data, with updated header and affine information,
                   saved on disk at a location determined by the mice.get_folder() method.
    """
    # Store the mice applied on template for inspection on 3d slicer
    new_image = new_image = nib.Nifti1Image(new_volume, affine=template.affine)
    new_image.set_sform(template.get_nib().affine, code=1)
    new_image.header.set_zooms(mice.get_segmentations().get_nib().header.get_zooms())
    new_image.set_qform(mice.get_segmentations().get_nib().affine, code=1)

    nib.save(new_image, f"{mice.get_folder()}/NF_ua_test_aligned_volume.nii.gz")

    return new_image
'''