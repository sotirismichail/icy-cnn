
** Project Structure

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
└── README.md                   # Project overview and instructions,,.