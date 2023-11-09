# bilinear.py
import numpy as np

from config.settings import ftype
from config.settings import utype


def _coord_map(
    dim: int,
    coord: float,
    mode: str,
) -> int:
    """
    Maps a coordinate value to the correct location based on the interpolation mode.

    Args:
        dim (int): The dimension (width or height) of the image.
        coord (float): The original coordinate value.
        mode (str): The mode of interpolation ('W' for warp, 'M' for mirror).

    Returns:
        int: The mapped coordinate value.
    """

    coord = np.floor(coord).astype(int)
    dim = np.floor(dim).astype(int)
    if mode == "M":
        if coord < 0:
            coord = np.fmod(-coord, dim)
        elif coord == dim:
            coord = dim - 1
        else:
            coord = dim - np.fmod(coord, dim)
    elif mode == "W":
        if coord < 0:
            coord = dim - np.fmod(-coord, dim)
        elif coord == dim:
            coord = 0
        else:
            coord = np.fmod(coord, dim)

    return coord


def interp_bilinear(
    img_channel: np.ndarray,
    tf_coords_r: np.ndarray,
    tf_coords_c: np.ndarray,
    mode: str = "M",
    cval: int = 0,
) -> np.ndarray:
    """
    Performs bilinear interpolation on a single channel of an image.

    Args:
        img_channel (np.ndarray): The image channel to interpolate.
        tf_coords_r (np.ndarray): The row coordinates for interpolation.
        tf_coords_c (np.ndarray): The column coordinates for interpolation.
        mode (str): The mode of interpolation ('C' for constant, 'W' for warp, 'M' for
            mirror).
        cval (int): The constant value to use if mode is 'C'.

    Returns:
        np.ndarray: The interpolated image channel.
    """

    img_channel = img_channel.astype(utype)
    tf_coords_r = tf_coords_r.astype(ftype)
    tf_coords_c = tf_coords_c.astype(ftype)

    output = np.empty(tf_coords_r.shape, dtype=utype)

    rows, columns = img_channel.shape
    tf_rows, tf_columns = tf_coords_r.shape

    for tfr in range(tf_rows):
        for tfc in range(tf_columns):
            r = tf_coords_r[tfr, tfc]
            c = tf_coords_c[tfr, tfc]

            if (mode == "C") and ((r < 0) or (r >= rows) or (c < 0) or (c >= columns)):
                output[tfr, tfc] = cval
            else:
                r = _coord_map(rows, r, mode)
                c = _coord_map(columns, c, mode)

                r_int = np.floor(r).astype(int)
                c_int = np.floor(c).astype(int)

                t = r - r_int
                u = c - c_int

                y0 = img_channel[r_int, c_int]
                y1 = img_channel[_coord_map(rows, r_int + 1, mode), c_int]
                y2 = img_channel[
                    _coord_map(rows, r_int + 1, mode),
                    _coord_map(columns, c_int + 1, mode),
                ]
                y3 = img_channel[r_int, _coord_map(columns, c_int + 1, mode)]

                output[tfr, tfc] = (
                    (1 - t) * (1 - u) * y0
                    + t * (1 - u) * y1
                    + t * u * y2
                    + (1 - t) * u * y3
                )

    return output


if __name__ == "__main__":
    # Example usage:
    try:
        # Load an image using your preferred method
        # image = load_image('path_to_image.jpg')
        # Define your coordinates (coords_r and coords_c)
        # Perform interpolation
        # interpolated_image = interp_bilinear(image[..., 0], coords_r, coords_c)
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
