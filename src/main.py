# main.py
import argparse

from display import display_images
from image_loader import load_image
from padding import pad
from transforms import log_cordic_transform
from transforms import logpolar


def main(image_path, padding_size):
    # Load the original image
    original_image = load_image(image_path)

    # Get dimensions for the log-polar transformation of the original image
    original_dimensions = original_image.shape[:2]

    # Apply the log-polar transform to the original image
    log_polar_image = logpolar(original_image, output_dim=original_dimensions)

    # Apply the CORDIC-based log-polar transform to the original image
    log_cordic_image = log_cordic_transform(
        original_image, output_dim=original_dimensions
    )

    # Apply cylindrical padding to the original image
    padded_original = pad(original_image, padding_size, mode="cylindrical")

    # Get dimensions for the log-polar transformation of the padded image
    padded_dimensions = padded_original.shape[:2]

    # Apply the log-polar transform to the padded image
    padded_log_polar = logpolar(padded_original, output_dim=padded_dimensions)
    padded_log_cordic = log_cordic_transform(
        padded_original, output_dim=padded_dimensions
    )

    # Display all the images
    images = [
        original_image,
        log_polar_image,
        log_cordic_image,
        padded_original,
        padded_log_polar,
        padded_log_cordic,
    ]
    titles = [
        "Original image",
        "Log-polar Transform",
        "Log-CORDIC Transform",
        "Cylindrical padding on original image",
        "Log-polar Transform after padding",
        "Log-CORDIC Transform after padding",
    ]
    display_images(images, titles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    parser.add_argument(
        "-p", "--padding", type=int, default=0, help="Padding size as an integer"
    )

    args = parser.parse_args()

    main(args.image, (args.padding, args.padding))
