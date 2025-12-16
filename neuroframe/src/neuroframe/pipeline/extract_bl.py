# ================================================================
# 0. Section: Imports
# ================================================================
import cv2

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from ..utils import get_z_coord
from ..logger import logger
from ..registrator import Registrator, SUTURE_REGISTRATOR, convert_input, apply_shape
from ..mouse import Mouse


# ──────────────────────────────────────────────────────
# 0.1 Subsection: Universal Constants
# ──────────────────────────────────────────────────────
SUTURE_TEMPLATE = cv2.imread("src/neuroframe/templates/suture_template_t14.png", cv2.IMREAD_GRAYSCALE)
BREGMA_TEMPLATE = convert_input(cv2.imread("src/neuroframe/templates/bregma_template_t14.png", cv2.IMREAD_GRAYSCALE))
LAMBDA_TEMPLATE = convert_input(cv2.imread("src/neuroframe/templates/lambda_template_t14.png", cv2.IMREAD_GRAYSCALE))
REF_TEMPLATES = (BREGMA_TEMPLATE, LAMBDA_TEMPLATE)



# ================================================================
# 1. Section: Extract Bregma and Lambda Points
# ================================================================
def get_bregma_lambda(mouse: Mouse, skull_surface: np.ndarray):
    transform = extract_deformation_map(skull_surface)

    # Unpacks the templates
    bregma_template, lambda_template = REF_TEMPLATES
    reference_slide = convert_input(mouse.segmentation.volume[100,:,:])
    bregma_template = apply_shape(reference_slide, bregma_template)
    lambda_template = apply_shape(reference_slide, lambda_template)

    # Get the bregma and lambda coordinates (x, y)
    bregma_coords = np.round(get_reference_point(bregma_template, skull_surface, transform)).astype(int)
    lambda_coords = np.round(get_reference_point(lambda_template, skull_surface, transform)).astype(int)
    
    # Get the z coordinates
    bregma_z = get_z_coord(mouse.micro_ct.data, bregma_coords)
    lambda_z = get_z_coord(mouse.micro_ct.data, lambda_coords)

    # Get the coordinates (z, y, x)
    bregma_coords = (bregma_z, bregma_coords[0], bregma_coords[1])
    lambda_coords = (lambda_z, lambda_coords[0], lambda_coords[1])

    # Log the coordinates
    logger.info(f"Bregma Coordinates: {bregma_coords} (z, y, x)")
    logger.info(f"Lambda Coordinates: {lambda_coords} (z, y, x)")

    # Log the deviations and angle
    deviations, angle = compute_deviation(mouse, (bregma_coords, lambda_coords))
    logger.info(f"Deviation Bregma {deviations[0].round(1)} mm")
    logger.info(f"Deviation Lambda {deviations[1].round(1)} mm")
    logger.info(f"Angle: {angle.round(2)} degrees")

    return bregma_coords, lambda_coords


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Deformation Map Extraction
# ──────────────────────────────────────────────────────
def extract_deformation_map(skull_surface: np.ndarray, sutures_registration: None | Registrator = None):
    # Initilize the sutures registrator if not provided
    if sutures_registration is None: sutures_registration = SUTURE_REGISTRATOR
        
    # Bspline registration to the suture template
    _, sutures_transform = sutures_registration.register(skull_surface, SUTURE_TEMPLATE)

    logger.detail(f"Obtained Transform Parameters: {sutures_transform.GetParameters()}")

    return sutures_transform


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Apply Deformation Map to Reference Templates
# ──────────────────────────────────────────────────────
def get_reference_point(reference_template: np.ndarray | sitk.Image, skull_surface: np.ndarray | sitk.Image, transform: sitk.Transform):
    # Apply the same transformation to the reference template that was used to get the template
    resampler = Registrator(res_interpolator = 'nearest')
    deformed_template = resampler.resample(convert_input(skull_surface), convert_input(reference_template), transform)
    deformed_template = sitk.GetArrayFromImage(deformed_template)

    # Get the coordinates of the reference points
    points = np.argwhere(deformed_template > 0)
    reference_coords = np.mean(points, axis=0)
    
    return reference_coords


def compute_deviation(mouse: Mouse, coords: tuple):
    bregma, lambda_ = coords
    midline = np.array(mouse.data_shape) // 2
    voxel_size = mouse.voxel_size

    midline_x = midline[2]
    bregma_x = bregma[2]
    lambda_x = lambda_[2]

    deviation_bregma = (midline_x - bregma_x) * voxel_size[2]
    deviation_lambda = (midline_x - lambda_x) * voxel_size[2]

    vector = np.array(bregma) - np.array(lambda_)
    vector = np.array([vector[2], vector[1]])
    vector = vector / np.linalg.norm(vector)
    print(vector)
    reference = np.array([0,1])

    # Calculate the angle between the vector and the reference
    angle = np.arccos(np.clip(np.dot(vector, reference), -1.0, 1.0))
    angle = 180 - np.degrees(angle)
    deviations = np.array([deviation_bregma, deviation_lambda])

    return deviations, angle


def inspect_bl(skull_surface: np.array, bregma_coords: np.array, lambda_coords: np.array):
    plt.figure()
    plt.imshow(skull_surface, cmap="gray")
    plt.scatter(bregma_coords[2], bregma_coords[1], c="red", marker='x', s=15, label=f"Bregma (z={bregma_coords[0]})")
    plt.scatter(lambda_coords[2], lambda_coords[1], c="blue", marker='x', s=15, label=f"Lambd (z={lambda_coords[0]})")
    plt.title("Transformed Bregma Template")
    plt.legend()
    plt.axis("off")

    plt.tight_layout
    plt.show()