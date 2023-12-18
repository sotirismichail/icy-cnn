import time

import numpy as np
from scipy import ndimage
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.transforms import logpolar


# Function to load and preprocess CIFAR-10 data
def load_cifar10_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Transform the images using logpolar
    transformed_train_images = np.array(
        [logpolar(img, (32, 32)) for img in train_images]
    )

    # Reshape data for the model
    train_images = transformed_train_images.reshape((-1, 32, 32, 3))
    test_images = test_images.reshape((-1, 32, 32, 3))

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


# Function to create and compile the model
def create_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Function to train and evaluate the model, and create a report
def train_and_evaluate_cifar10():
    start_time = time.time()

    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()

    # Create and compile the model
    model = create_model()

    # Train the model
    train_start_time = time.time()
    model.fit(train_images, train_labels, epochs=5)
    train_end_time = time.time()

    # Apply random rotations to the test images
    rotated_test_images = np.array(
        [
            ndimage.rotate(
                img.squeeze(), angle=np.random.uniform(-180, 180), reshape=False
            )
            for img in test_images
        ]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(rotated_test_images, test_labels)
    test_end_time = time.time()

    # Write the report to a file
    timestamp = time.strftime("%d%m%Y_%H%M")
    with open(f"test_run_{timestamp}.txt", "w") as file:
        file.write(
            f"Total Training Time: {train_end_time - train_start_time:.2f} seconds\n"
        )
        file.write(
            f"Total Testing Time: {test_end_time - train_end_time:.2f} seconds\n"
        )
        file.write(f"Total Time: {test_end_time - start_time:.2f} seconds\n")
        file.write(f"Test Accuracy: {test_acc:.4f}\n")


# Execute the training and evaluation
train_and_evaluate_cifar10()
