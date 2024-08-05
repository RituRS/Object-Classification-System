# Object Classification System

## Overview

The Object Classification System is a web application built with Streamlit that allows users to upload an image and classify it into one of several categories using advanced deep learning models. The application supports two models: CNN and MobileNetV2. It also provides detailed performance metrics including confusion matrix, precision-recall curve, ROC curve, training and validation metrics, weighted averages, and dataset distribution.

## Features

- **Image Upload**: Upload an image to classify it using the selected model.
- **Model Selection**: Choose between CNN and MobileNetV2 for classification.
- **Confidence Scores**: View confidence scores for each class with horizontal scrolling.
- **Performance Metrics**: Visualize various performance metrics for the selected model.
- **Visualization Options**: Choose from confusion matrix, precision-recall curve, ROC curve, training metrics, weighted averages, and dataset distribution.

## Requirements

To run this application, you need the following:

- Python 3.x
- Streamlit
- TensorFlow
- Pillow
- Numpy

You can install the required Python packages using pip:

```bash
pip install streamlit tensorflow pillow numpy

## Setup
Clone the Repository:
git clone https://github.com/RituRS/Object-Classification-System.git
cd object-classification-system
Place the Models:
Ensure that the trained model files (cnn_model.h5 and mobilenet_model.h5) are located in the root directory of the project.

Run the Streamlit App:
streamlit run app.py

## Usage
Upload an Image: Click on "Upload an image..." and select an image file to classify.
Select a Model: Choose between "CNN" and "MobileNetV2" from the model dropdown.
View Predictions: After uploading the image, the application will display the predicted class along with confidence scores for each class.
Visualize Performance Metrics: Use the dropdown to select the type of visualization you want to see, such as confusion matrix, precision-recall curve, ROC curve, etc.

Example
Here is a sample output after uploading an image and selecting a model:

Model Prediction: Dog
Confidence Scores: (Scroll horizontally to view scores for all classes)

Acknowledgements
The CIFAR-10 dataset used for evaluation.
TensorFlow and Streamlit for the powerful tools used in this application.
