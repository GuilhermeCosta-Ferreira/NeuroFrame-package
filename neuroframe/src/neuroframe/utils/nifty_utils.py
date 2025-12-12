# ================================================================
# 0. Section: Imports
# ================================================================
import nibabel as nib



# ================================================================
# 1. Section: Convert Files
# ================================================================
def compress_nifty(input_path: str, output_path: str) -> None:
    img = nib.load(input_path)

    # Save the NIfTI file with gzip compression
    nib.save(img, output_path)