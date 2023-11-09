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

    n, m = matrix.shape
    r, c = padding

    matrix_padded = np.zeros((n + r * 2, m + c * 2), dtype=utype)
    matrix_padded[r : n + r, c : m + c] = matrix

    if mode == "cylindrical":
        matrix_padded[:r, c : m + c] = matrix[-r:, :]
        matrix_padded[n + r :, c : m + c] = matrix[:r, :]
    elif mode != "zeros":
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
