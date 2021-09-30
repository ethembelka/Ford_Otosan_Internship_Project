import os
# Path to jsons
JSON_DIR = '../input/forstajp1/p1/p1/intern-p1/data/jsons'

# Path to mask
MASK_DIR  = '../input/forstajp1/masks/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = './masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = '../input/forstajp1/p1/p1/intern-p1/data/images'


# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = True

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2