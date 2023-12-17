# mnist.py
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.transforms import log_cordic_transform

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Transform the images using log_cordic_transform
transformed_train_images = np.array(
    [log_cordic_transform(img, (28, 28)) for img in train_images]
)
transformed_test_images = np.array(
    [log_cordic_transform(img, (28, 28)) for img in test_images]
)

# Reshape data for the model
train_images = transformed_train_images.reshape((-1, 28, 28, 1))
test_images = transformed_test_images.reshape((-1, 28, 28, 1))

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
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
