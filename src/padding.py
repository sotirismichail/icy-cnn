# padding.py
from typing import Tuple

import numpy as np

from config.settings import utype


def pad(
    matrix: np.ndarray, padding: Tuple[int, int], mode: str = "zeros"
) -> np.ndarray:
    """Pad a matrix with cylindrical padding.

    To achieve cylindrical convolution, instead of padding the matrix with zeroes, the
    elements of the first and last rows are copied to the opposite ends of the matrix.
    Left and right padding are filled with zeroes.

    Args:
        matrix (np.ndarray): The matrix to pad.
        padding (Tuple[int, int]): The amount of padding for rows and columns.
        mode (str): The padding mode, either "zeros" or "cylindrical".

    Returns:
        np.ndarray: The padded matrix.

    Raises:
        ValueError: If an unsupported padding mode is provided.
    """

    if len(matrix.shape) == 3:
        # Handle 3D array (RGB image)
        n, m, channels = matrix.shape
        matrix_padded = np.zeros(
            (n + padding[0] * 2, m + padding[1] * 2, channels), dtype=utype
        )
        matrix_padded[
            padding[0] : n + padding[0], padding[1] : m + padding[1], :
        ] = matrix

        if mode == "cylindrical":
            matrix_padded[: padding[0], padding[1] : m + padding[1], :] = matrix[
                -padding[0] :, :, :
            ]
            matrix_padded[n + padding[0] :, padding[1] : m + padding[1], :] = matrix[
                : padding[0], :, :
            ]
    elif len(matrix.shape) == 2:
        # Handle 2D array (grayscale image)
        n, m = matrix.shape
        matrix_padded = np.zeros((n + padding[0] * 2, m + padding[1] * 2), dtype=utype)
        matrix_padded[padding[0] : n + padding[0], padding[1] : m + padding[1]] = matrix

        if mode == "cylindrical":
            matrix_padded[: padding[0], padding[1] : m + padding[1]] = matrix[
                -padding[0] :, :
            ]
            matrix_padded[n + padding[0] :, padding[1] : m + padding[1]] = matrix[
                : padding[0], :
            ]
    else:
        raise ValueError(f"Unsupported matrix shape: {matrix.shape}")

    if mode not in ["zeros", "cylindrical"]:
        # Raise an error for unsupported padding modes.
        raise ValueError(f"Unsupported padding mode: {mode}")

    return matrix_padded


# Example usage:
if __name__ == "__main__":
    # Example matrix and padding values.
    example_matrix = np.array([[1, 2], [3, 4]], dtype=utype)
    example_padding = (1, 1)

    # Apply cylindrical padding.
    padded_matrix = pad(example_matrix, example_padding, mode="cylindrical")
    print(padded_matrix)
