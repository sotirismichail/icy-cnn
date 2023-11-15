# transforms.py
import numpy as np

from src.bilinear import interp_bilinear
from src.cordic import cordic_vector_mode


def logpolar(
    image: np.ndarray, output_dim: tuple[int, int], mode: str = "M", cval: int = 0
) -> np.ndarray:
    """
    Applies the log-polar transform to an image with the specified output dimensions.

    Args:
        image (np.ndarray): The input image to transform.
        output_dim (tuple[int, int]): The dimensions (height, width) of the output image
        mode (str): How values outside the borders are handled. 'C' for constant,
                    'M' for mirror, 'W' for wrap.
        cval (int): Constant to fill the outside area with if mode is 'C'.

    Returns:
        np.ndarray: The log-polar transformed image.
    """

    angles = output_dim[1]  # Width of the output image corresponds to number of angles
    radius = output_dim[0]  # Height of the output image corresponds to number of radii

    # Center of the log-polar coordinates should be the center of the image
    centre = (np.array(image.shape[:2]) - 1) / 2.0

    # Calculate the maximum radius for the log-polar space based on the input image
    # dimensions
    max_radius = np.hypot(*(centre))

    # Calculate log_base using the desired number of radii and the maximum radius
    # This is the change in radius per pixel in the log-polar space
    log_base = np.log(max_radius) / (
        radius - 1
    )  # Corrected to use the desired number of radii

    # Calculate the linear steps in the logarithmic space for radii using log_base
    log_rad = log_base * np.arange(radius)  # This is where log_base is used

    angles_arr = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    # Use broadcasting to create a 2D array for theta and log radius
    theta, log_r = np.meshgrid(angles_arr, log_rad)

    # Convert log radius back to linear radius
    r = np.exp(log_r)

    # Convert polar coordinates to Cartesian coordinates
    coords_r = r * np.sin(theta) + centre[0]
    coords_c = r * np.cos(theta) + centre[1]

    # Initialize output image with the specified output dimensions
    output = np.zeros((radius, angles, image.shape[2]), dtype=image.dtype)

    # Perform bilinear interpolation
    for channel in range(image.shape[2]):
        output[..., channel] = interp_bilinear(
            image[..., channel], coords_r, coords_c, mode=mode, cval=cval
        )

    return output.squeeze()


def log_cordic_transform(
    image: np.ndarray, output_dim: tuple[int, int], mode: str = "M", cval: int = 0
) -> np.ndarray:
    """
    Applies the log-polar transform to an image with the specified output dimensions using the CORDIC algorithm.

    Args:
        image (np.ndarray): The input image to transform.
        output_dim (tuple[int, int]): The dimensions (height, width) of the output image.
        mode (str): The mode of interpolation ('C' for constant, 'W' for warp, 'M' for mirror).
        cval (int): The constant value to use if mode is 'C'.

    Returns:
        np.ndarray: The log-polar transformed image.
    """

    angles = output_dim[1]
    radius = output_dim[0]
    centre = (np.array(image.shape[:2]) - 1) / 2.0
    max_radius = np.hypot(*(centre))
    log_base = np.log(max_radius) / (radius - 1)

    # Initialize output image
    output = np.zeros((radius, angles, image.shape[2]), dtype=image.dtype)

    tf_coords_r = np.zeros((radius, angles))
    tf_coords_c = np.zeros((radius, angles))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Calculate coordinates relative to the center
            rel_x, rel_y = x - centre[1], y - centre[0]

            # Apply CORDIC to get polar coordinates
            cordic_radius, cordic_angle = cordic_vector_mode(rel_x, rel_y)

            # Convert radius to log scale and map to output dimensions
            log_r = np.log(cordic_radius) / log_base if cordic_radius > 0 else 0
            polar_x, polar_y = int(cordic_angle * angles / (2 * np.pi)), int(
                log_r * radius / np.log(max_radius)
            )

            if 0 <= polar_x < angles and 0 <= polar_y < radius:
                tf_coords_r[polar_y, polar_x] = y
                tf_coords_c[polar_y, polar_x] = x

    # Initialize output image
    output = np.zeros((radius, angles, image.shape[2]), dtype=image.dtype)

    # Apply bilinear interpolation for each channel
    for channel in range(image.shape[2]):
        output[..., channel] = interp_bilinear(
            image[..., channel], tf_coords_r, tf_coords_c, mode=mode, cval=cval
        )

    return output.squeeze()
