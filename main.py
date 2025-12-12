from neuroframe.src.neuroframe import *
import time
import numpy as np

compress_nifty("../p324_uCT_reshape.nii", 
               "../compressed.nii.gz",
               data_compression=True)

'''
start_time = time.time()
mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
print("Loaded in --- %s seconds ---" % (time.time() - start_time), end="\n\n")

mouse.plot_segmentations_overlay()
mouse.plot_multimodal_midplanes()
'''