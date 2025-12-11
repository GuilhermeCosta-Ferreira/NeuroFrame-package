from neuroframe.src.neuroframe import *
import time
import numpy as np


start_time = time.time()
test = Mouse("P324", "../p324_mri.nii.gz", "../p324_uCT.nii.gz", "../p324_seg.nii.gz")
print(test)
print("Loaded in --- %s seconds ---" % (time.time() - start_time), end="\n\n")

start_time = time.time()
print(test.voxel_size)
print("Loaded in --- %s seconds ---" % (time.time() - start_time), end="\n\n")

test.plot_segmentations_overlay(slice_offset=50)
test.plot_multimodal_midplanes(slice_offset=50)