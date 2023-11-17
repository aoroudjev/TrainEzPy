# TrainEzPy:
The basic builder of basic object detectors.

#### Introduction

The module primarily leverages TensorFlow, Keras, and OpenCV libraries to facilitate the loading and processing of image data, including handling XML annotations, image transformations, and object detection using YOLOv8.

#### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- Keras-CV
- OpenCV
- NumPy
- tqdm
- Pillow

#### Installation
1. **Install Python ^3.8**: Download from [official Python website](https://www.python.org/downloads/).
2. **Install required libraries**: Use pip or poetry to install the necessary libraries.
```bash
pip install tensorflow keras keras-cv opencv-python numpy tqdm pillow
```

#### Usage
1. **Data Preparation**: Functions to load and preprocess images and their annotations.
    
    - `load_data_from_dir()`: Load images and XML annotations.
    - `load_image()`: Load a single image file.
    - `replace_white_background()`: Replace white background in images.
2. **Dataset Preparation**: Create TensorFlow datasets.
    
    - `load_dataset()`: Prepare datasets for training and validation.
3. **Model Training and Evaluation**: Build and train the YOLOv8 model.
    
    - Use `keras_cv.models.YOLOV8Detector` for the model.
    - Training process with `model.fit()`.
4. **Visualization**: Functions to visualize datasets and detection results.
    
    - `visualize_dataset()`
    - `visualize_detections()`
5. **Custom Image Testing**: Test the model on custom images.

#### Additional Information

- The module assumes the presence of image data and corresponding XML annotations in predefined directories.
- The module uses a YOLOv8 model with a custom backbone for object detection.
- For further customizations and advanced usage, refer to TensorFlow and Keras-CV documentation.
