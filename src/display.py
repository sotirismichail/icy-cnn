# display.py
import matplotlib.pyplot as plt
import numpy as np


def display_images(
    images: list[np.ndarray],
    titles: list[str],
) -> None:
    """Display a list of images with corresponding titles.

    Args:
        images (list): A list of np.ndarray images to display.
        titles (list): A list of titles for the images.

    Raises:
        ValueError: If the lengths of images and titles lists do not match.
    """

    if len(images) != len(titles):
        raise ValueError("The number of images and titles must match")

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    if num_images == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Example images and titles.
    example_images = [np.random.rand(100, 100), np.random.rand(100, 100)]
    example_titles = ["Image 1", "Image 2"]

    # Display the images.
    display_images(example_images, example_titles)
