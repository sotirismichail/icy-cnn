import time
from datetime import datetime

import numpy as np
from scipy import ndimage
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.transforms import logpolar


def train_and_evaluate_mnist():
    start_time_total = time.time()

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and reshape the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Transform the training images
    transformed_train_images = np.array(
        [logpolar(img, (28, 28)) for img in train_images]
    )

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Create the model
    model = Sequential(
        [
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    start_time_train = time.time()
    model.fit(transformed_train_images, train_labels, epochs=5)
    end_time_train = time.time()

    # Randomly rotate test images
    rotated_test_images = np.array(
        [
            ndimage.rotate(
                img.squeeze(), angle=np.random.uniform(-180, 180), reshape=False
            )
            for img in test_images
        ]
    )

    # Evaluate the model
    start_time_test = time.time()
    test_loss, test_acc = model.evaluate(rotated_test_images, test_labels)
    end_time_test = time.time()

    end_time_total = time.time()

    # Compile the report
    report = {
        "Training Time (seconds)": end_time_train - start_time_train,
        "Testing Time (seconds)": end_time_test - start_time_test,
        "Total Time (seconds)": end_time_total - start_time_total,
        "Test Accuracy": test_acc,
        "Test Loss": test_loss,
    }

    # Save the report to a file
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    filename = f"test_run_{timestamp}.txt"
    with open(filename, "w") as file:
        for key, value in report.items():
            file.write(f"{key}: {value}\n")

    print(f"Report saved to {filename}")


train_and_evaluate_mnist()
