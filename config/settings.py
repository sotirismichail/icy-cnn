# settings.py
import numpy as _np

# Define the data type for image processing.
# These can be changed according to the precision required.
# For instance, you might use np.float32 for faster computation
# but with less precision compared to np.float64.
ftype = _np.float64
itype = _np.int32
utype = _np.uint8

# Define the number of iterations for the CORDIC algorithm.
N_ITER = 16

# Define the path to the image if you want to have a default image to load.
IMAGE_PATH = "notebooks/doge.png"  # Replace with a valid image path
PADDING_SIZE = (10, 10)

# Add any other global constants or configuration settings that might be needed.
# For example, if you have settings for logging, API keys, etc., they would go here.
