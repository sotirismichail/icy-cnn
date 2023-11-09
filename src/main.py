# main.py
from config.settings import IMAGE_PATH
from config.settings import PADDING_SIZE
from display import display_images
from image_loader import load_image
from padding import pad
from transforms import log_cordic_transform
from transforms import logpolar


def main():
    # Load the original image
    original_image = load_image(IMAGE_PATH)

    # Apply the log-polar transform
    log_polar_image, _, _ = logpolar(original_image)

    # Apply the CORDIC-based log-polar transform
    cordic_log_polar_image = log_cordic_transform(original_image)

    # Apply cylindrical padding to the original image
    padded_original = pad(original_image, PADDING_SIZE, mode="cylindrical")

    # Apply cylindrical padding to the log-polar image
    padded_log_polar = pad(log_polar_image, PADDING_SIZE, mode="cylindrical")

    # Apply cylindrical padding to the CORDIC-log-polar image
    padded_cordic_log_polar = pad(
        cordic_log_polar_image, PADDING_SIZE, mode="cylindrical"
    )

    # Display all the images
    images = [
        original_image,
        log_polar_image,
        cordic_log_polar_image,
        padded_original,
        padded_log_polar,
        padded_cordic_log_polar,
    ]
    titles = [
        "Original Image",
        "Log-Polar Transform",
        "CORDIC-Log-Polar Transform",
        "Padded Original Image",
        "Padded Log-Polar Transform",
        "Padded CORDIC-Log-Polar Transform",
    ]
    display_images(images, titles)


if __name__ == "__main__":
    main()
