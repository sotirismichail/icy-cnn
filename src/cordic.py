from typing import Tuple
from typing import Union

import numpy as np

from config.settings import N_ITER


def cordic(
    x: float, y: float, theta: float, n: int = N_ITER, mode: str = "vector"
) -> Union[Tuple[float, float, float], Tuple[float, float]]:
    """
    Performs the CORDIC (COordinate Rotation DIgital Computer) algorithm
    either in vector mode or rotation mode.

    The CORDIC algorithm is used to calculate trigonometric functions and
    is suitable for hardware implementation.

    Args:
        x (float): The real component input (not used in vector mode).
        y (float): The imaginary component input (not used in vector mode).
        theta (float): The angle input (-pi to pi scaled to signed 16 bits,
                       not used in rotation mode).
        n (int): Number of iterations.
        mode (int): Operation mode (0 for rotation mode, 1 for vector mode).

    Returns:
        Tuple[float, float, float]: A tuple containing the real component output,
                                    imaginary component output, and angle output
                                    if in rotation mode
        Tuple[float, float]: A tuple containing the sine and the cosine of a given
                            angle, if in vector mode
    """

    K = 1.0
    z = theta

    def update_values(
        x: float, y: float, z: float, w: int
    ) -> Tuple[float, float, float, float]:
        """Updates the values of x, y, z, and K for each iteration."""
        d = -1 if z < 0 else 1
        x_next = x - y * d * 2**-w
        y_next = y + x * d * 2**-w
        z_next = z - d * np.arctan(2**-w)
        K_next = 1 / np.sqrt(1 + 2 ** (-2 * w))

        return x_next, y_next, z_next, K_next

    if mode == "vector":
        if np.pi / 2 < z <= np.pi or -np.pi <= z < -np.pi / 2:
            x, y, z = -x, -y, z - np.sign(z) * np.pi
        x, y = 1.0, 0.0

        for w in range(
            1, n
        ):  # Start at 1 since the first iteration doesn't change the value
            x, y, z, K_next = update_values(x, y, z, w)
            K *= K_next  # Multiply with the cumulative K factor

        x_out, y_out = K * x, K * y

        return y_out, x_out

    else:
        d = -1 if y >= 0 else 1
        x, y, z = -d * y, d * x, d * np.pi / 2

        for w in range(n):
            x, y, z, K = update_values(x, y, z, w)

        x_out, y_out = K * x, K * y

    return x_out, y_out, z
