# image_loader.py
from typing import Optional

import numpy as np
from PIL import Image


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from a file path and convert it to a NumPy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Optional[np.ndarray]: The loaded image as a NumPy array or None if an error
            occurs.

    Raises:
        FileNotFoundError: If the file at the specified path was not found.
        IOError: If there was an error opening the file at the specified path.
        Exception: If an unexpected error occurred.
    """

    try:
        with Image.open(image_path) as img:
            return np.array(img)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


if __name__ == "__main__":
    test_image_path = "path_to_test_image.jpg"
    test_image = load_image(test_image_path)
    if test_image is not None:
        print(f"Loaded image shape: {test_image.shape}")
    else:
        print("Failed to load the image.")
