# ================================================================
# 1. Section: Property Assertions
# ================================================================
def assert_folder_consitency(paths_dict: dict[str, str]) -> None:
    # Get's all the folder's
    micro_ct_folder = paths_dict["ct_path"]
    mri_folder = paths_dict["mri_path"]
    segmentations_folder = paths_dict["segmentations_path"]

    # Check if they all match
    if not (micro_ct_folder == mri_folder == segmentations_folder): print("Warning: The folder paths of the mouse data do not match, defaulted to MRI folder path.")

def assert_shape_consitency(shapes: list[tuple[int, int, int]]) -> None:
    # Check if all shapes match
    if not all(shape == shapes[0] for shape in shapes): print("Warning: The shapes of the mouse data do not match, defaulted to MRI shape.")

def assert_voxel_size_consitency(voxel_sizes: list[tuple[float, float, float]]) -> None:
    # Check if all voxel sizes match
    if not all(voxel_size == voxel_sizes[0] for voxel_size in voxel_sizes): print("Warning: The voxel sizes of the mouse data do not match, defaulted to MRI voxel size.")