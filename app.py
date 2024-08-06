import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Define model file paths
cnn_model_path = 'cnn_model.h5'
mobilenet_model_path = 'mobilenet_model.h5'

# Function to download models
def download_model(url, destination):
    if not os.path.exists(destination):
        gdown.download(url, destination, quiet=True)

# Download models if not present
download_model('https://drive.google.com/uc?id=1LyVJ_ZkvSXWaDHqYZRCRb5fpI8_ahAII', cnn_model_path)
download_model('https://drive.google.com/uc?id=1WbjemJD0ckfTIvnT0Jx8qxiE7z-xCjAU', mobilenet_model_path)

# Load trained models
cnn_model = tf.keras.models.load_model(cnn_model_path)
mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)

# Class names for CIFAR-10 dataset
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Define model dictionary
models = {
    'CNN': cnn_model,
    'MobileNetV2': mobilenet_model
}

# Predict function
def predict(model, image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence_scores = predictions[0]
    return predicted_class, confidence_scores

# Streamlit app
st.title("Object Classification System")
st.write("Welcome to the Object Classification System! This tool enables you to upload an image and classify it into one of several categories of objects using advanced models. You can also visualize performance metrics for the selected model to understand its accuracy and reliability.")

# Dropdown for model selection
selected_model_name = st.selectbox("Select a model to view performance metrics:", list(models.keys()))

# Dropdown for visualization type
visualization_type = st.selectbox("Select a visualization type:", ["Confusion Matrix", "Precision-Recall Curve", "ROC Curve", "Training and Validation Metrics", "Weighted Averages", "Dataset Distribution"])

uploaded_file = st.file_uploader("Upload an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    st.write("Classifying the image using the selected model...")

    selected_model = models[selected_model_name]
    if selected_model:
        prediction, confidence_scores = predict(selected_model, image)
        st.write(f"**{selected_model_name} Model Prediction:** {prediction}")
        
        # Display confidence scores in a horizontally scrolling container
        st.write("**Confidence Scores:**")
        st.markdown(
            f"""
            <div style="overflow-x: auto; white-space: nowrap;">
                {" ".join([f"<div style='display: inline-block; padding: 10px;'>{class_name}: {score:.4f}</div>" for class_name, score in zip(class_names, confidence_scores)])}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write(f"Model {selected_model_name} could not be loaded.")

if st.button("Show Visualization"):
    st.write(f"### Visualization for {selected_model_name}")

    # Define paths for saved images using relative paths
    if selected_model_name == 'CNN':
        prefix = 'cnn_model_'
    else:
        prefix = 'mobilenetv2_model_'

    # Paths
    paths = {
        "Confusion Matrix": f'{prefix}confusion_matrix.png',
        "Precision-Recall Curve": f'{prefix}precision_recall_curve.png',
        "ROC Curve": f'{prefix}roc_curve.png',
        "Training and Validation Metrics": f'{prefix}metrics_plot.png',
        "Weighted Averages": f'{prefix}weighted_averages.png',
        "Dataset Distribution": 'dataset_distribution.png'
    }

    # Display the selected visualization
    if visualization_type in paths:
        path = paths[visualization_type]
        if os.path.isfile(path):
            st.image(path, caption=f"{selected_model_name} - {visualization_type}", use_column_width=True)
        else:
            st.write(f"{visualization_type} image not found. Please make sure the file is uploaded and try again.")
    else:
        st.write("Please select a valid visualization type.")

st.write("### About")
st.write(
    "The Object Classification System leverages deep learning models to classify images into predefined categories. "
    "The app uses two advanced models: CNN and MobileNetV2. After uploading an image, you can get predictions "
    "from the selected model and view detailed performance metrics including confusion matrix, precision-recall curve, ROC curve, training metrics, weighted averages of metrics, "
    "and dataset distribution. The CIFAR-10 dataset, which includes classes such as airplane, automobile, bird, cat, deer, "
    "dog, frog, horse, ship, and truck, is used for evaluation. Use the dropdowns to select the model and the type of "
    "visualization you want to see."
)
