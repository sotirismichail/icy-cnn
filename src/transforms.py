# transforms.py
import numpy as np

from .bilinear import interp_bilinear
from .cordic import cordic_vector_mode


def logpolar(
    image: np.ndarray, angles: int = None, mode: str = "M", cval: int = 0
) -> np.ndarray:
    """Applies the log-polar transform to an image.

    Args:
        image (np.ndarray): The input image to transform.
        angles (int, optional): Number of samples in the radial direction. Defaults to
            the larger of the image dimensions.
        mode (str, optional): How values outside the borders are handled. 'C' for
            constant, 'M' for mirror, 'W' for wrap. Defaults to 'M'.
        cval (int, optional): Constant to fill the outside area with if mode is 'C'.
            Defaults to 0.

    Returns:
        np.ndarray: The log-polar transformed image.
    """

    if angles is None:
        angles = max(image.shape[:2])

    centre = (np.array(image.shape[:2]) - 1) / 2.0
    d = np.hypot(*(image.shape[:2] - centre))
    log_base = np.log(d) / angles

    angles_arr = -np.linspace(0, 2 * np.pi, 2 * angles + 1)[:-1]
    theta = np.empty((len(angles_arr), angles), dtype=image.dtype)
    theta.T[:] = angles_arr
    log_e = np.arange(angles, dtype=image.dtype)
    r = np.exp(log_e * log_base)

    coords_r = r * np.sin(theta) + centre[0]
    coords_c = r * np.cos(theta) + centre[1]

    channels = image.shape[2]
    output = np.empty(coords_r.shape + (channels,), dtype=image.dtype)

    for channel in range(channels):
        output[..., channel] = interp_bilinear(
            image[..., channel], coords_r, coords_c, mode=mode, cval=cval
        )

    return output.squeeze()


def log_cordic_transform(image: np.ndarray) -> np.ndarray:
    """Applies the CORDIC-based log-polar transform to an image.

    Args:
        image (np.ndarray): The input image to transform.

    Returns:
        np.ndarray: The CORDIC-log-polar transformed image.
    """

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    transformed_image = np.zeros_like(image)
    max_radius = np.sqrt(cx**2 + cy**2)

    for i in range(h):
        for j in range(w):
            log_r = np.exp(j / w * np.log(max_radius))
            theta = (i / h) * 2 * np.pi
            x, _ = cordic_vector_mode(log_r, theta)
            y = cy - log_r * np.sin(theta)

            if 0 <= x < w and 0 <= y < h:
                transformed_image[i, j] = image[int(y), int(x)]

    return transformed_image
