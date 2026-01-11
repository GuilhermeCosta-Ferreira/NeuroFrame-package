# ================================================================
# 0. Section: Imports
# ================================================================
from .graphic_utils import transparent_to_color_cmap
from .colors import *


# ================================================================
# 1. Section: Colormaps
# ================================================================
alpha_red_cmap_256 = transparent_to_color_cmap(NR_RED)
alpha_blue_cmap_256 = transparent_to_color_cmap(NR_BLUE)

alpha_black_red_cmap_256 = transparent_to_color_cmap(NR_RED, '#000000')
alpha_black_blue_cmap_256 = transparent_to_color_cmap(NR_BLUE, '#000000')