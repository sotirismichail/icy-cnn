## Overview

This project implements an image processing pipeline that includes loading images, applying various transformations, padding, and displaying the results. It uses the CORDIC algorithm for certain transformations and provides a structured approach to image processing in Python.

## Project Structure
```
icy_cnn/
│
├── src/                        # Source files
│   ├── main.py                 # Entry point of the application
│   ├── image_loader.py         # Module to handle image loading
│   ├── transforms.py           # Module for different image transforms
│   ├── padding.py              # Module for image padding functions
│   ├── display.py              # Module for image display functions
│   └── cordic.py               # Module for CORDIC algorithm functions
│
├── config/                     # Configuration files
│   └── settings.py             # Global settings (e.g., image paths, constants)
│
├── utils/                      # Utility code
│   └── bilinear_interpolation.py # Utility functions for bilinear interpolation
│
├── tests/                      # Test suite for the project
│   └── test_transforms.py      # Tests for transform functions
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   └── experiment.ipynb        # A notebook for trying out ideas
│
├── requirements.txt            # Python dependencies required for the project
└── README.md                   # Project overview and instructions
```

## Installation

To run this project, you will need Python 3.8 or later. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/icy_cnn.git
cd icy_cnn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

To run the project, execute the `main.py` script from the `src` directory. For example,
transforming the "lion.jpeg" image with a 125 pixel padding, can be done as follows:

```bash
python src/main.py -i notebooks/lion.jpeg -p 125
```

This will load the image specified in the `IMAGE_PATH`, apply the transformations, padding, and display the results.

## Testing

To run the tests, navigate to the `tests` directory and execute the test scripts:

```bash
python -m unittest discover tests
```

## Jupyter Notebooks

For experimental purposes, Jupyter notebooks are provided in the `notebooks` directory. You can start the Jupyter notebook server with:

```bash
jupyter notebook notebooks/
```

Then open the `experiment.ipynb` notebook to try out different ideas interactively.
```