# cordic.py
import numpy as np

from config.settings import N_ITER

cordic_angles = np.arctan(2 ** -np.arange(N_ITER)).astype(np.float32)


def cordic_vector_mode(
    x0: float,
    y0: float,
    iterations: int = N_ITER,
) -> tuple[float, float]:
    """CORDIC algorithm in vector mode.

    Args:
        x0 (float): The initial x-coordinate.
        y0 (float): The initial y-coordinate.
        iterations (int, optional): Number of iterations to perform. Defaults to N_ITER.

    Returns:
        Tuple[float, float]: The transformed coordinates (xi, zi).
    """

    xi, yi, zi = x0, y0, 0
    for i in range(iterations):
        di = -1 if yi < 0 else 1
        xi_next = xi - yi * di * (2**-i)
        yi_next = yi + xi * di * (2**-i)
        zi_next = zi - di * cordic_angles[i]
        xi, yi, zi = xi_next, yi_next, zi_next
    return abs(xi), zi
