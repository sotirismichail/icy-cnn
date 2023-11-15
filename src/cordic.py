# cordic.py
import numpy as np

from config.settings import N_ITER


def cordic_vector_mode(
    x0: float,
    y0: float,
    iterations: int = N_ITER,
) -> tuple[float, float]:
    """
    CORDIC algorithm in vector mode for converting Cartesian coordinates to polar.

    Args:
        x0 (float): The initial x-coordinate.
        y0 (float): The initial y-coordinate.
        iterations (int): Number of iterations to perform. Defaults to 16.

    Returns:
        Tuple[float, float]: The polar coordinates (radius, angle).
    """

    # Initialize variables
    xi, yi, zi = x0, y0, 0.0
    angle_LUT = np.arctan(2.0 ** -np.arange(iterations))

    # CORDIC Iterations
    for i in range(iterations):
        di = -1 if yi < 0 else 1
        xi_next = xi - yi * di * (2**-i)
        yi_next = yi + xi * di * (2**-i)
        zi_next = zi - di * angle_LUT[i]

        xi, yi, zi = xi_next, yi_next, zi_next

    # Calculate radius and angle
    radius = np.sqrt(xi**2 + yi**2)
    angle = zi

    return radius, angle
