import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load trained models
cnn_model = tf.keras.models.load_model('cnn_model.h5')
mobilenet_model = tf.keras.models.load_model('mobilenet_model.h5')

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

    # Define paths for saved images using absolute paths
    base_path = 'C:\\Users\\RITUJA\\CIFAR10'
    if selected_model_name == 'CNN':
        prefix = 'cnn_model_'
    else:
        prefix = 'mobilenetv2_model_'

    confusion_matrix_path = os.path.join(base_path, f'{prefix}confusion_matrix.png')
    precision_recall_path = os.path.join(base_path, f'{prefix}precision_recall_curve.png')
    roc_curve_path = os.path.join(base_path, f'{prefix}roc_curve.png')
    training_metrics_path = os.path.join(base_path, f'{prefix}metrics_plot.png')
    weighted_averages_path = os.path.join(base_path, f'{prefix}weighted_averages.png')
    dataset_distribution_path = os.path.join(base_path, 'dataset_distribution.png')

    if visualization_type == "Confusion Matrix":
        try:
            st.image(confusion_matrix_path, caption=f"{selected_model_name} - Confusion Matrix", use_column_width=True)
        except FileNotFoundError:
            st.write("Confusion matrix image not found.")
        except Exception as e:
            st.write(f"An error occurred: {e}")

    elif visualization_type == "Precision-Recall Curve":
        try:
            st.image(precision_recall_path, caption=f"{selected_model_name} - Precision-Recall Curve", use_column_width=True)
        except FileNotFoundError:
            st.write("Precision-recall curve image not found.")
        except Exception as e:
            st.write(f"An error occurred: {e}")

    elif visualization_type == "ROC Curve":
        try:
            st.image(roc_curve_path, caption=f"{selected_model_name} - ROC Curve", use_column_width=True)
        except FileNotFoundError:
            st.write("ROC curve image not found.")
        except Exception as e:
            st.write(f"An error occurred: {e}")

    elif visualization_type == "Training and Validation Metrics":
        try:
            st.image(training_metrics_path, caption=f"{selected_model_name} - Training and Validation Metrics", use_column_width=True)
        except FileNotFoundError:
            st.write("Training metrics image not found.")
        except Exception as e:
            st.write(f"An error occurred: {e}")

    elif visualization_type == "Weighted Averages":
        try:
            st.image(weighted_averages_path, caption=f"{selected_model_name} - Weighted Averages of Metrics", use_column_width=True)
        except FileNotFoundError:
            st.write("Weighted averages plot image not found.")
        except Exception as e:
            st.write(f"An error occurred: {e}")

    elif visualization_type == "Dataset Distribution":
        try:
            st.image(dataset_distribution_path, caption="Dataset Distribution", use_column_width=True)
        except FileNotFoundError:
            st.write("Dataset distribution plot image not found.")
        except Exception as e:
            st.write(f"An error occurred: {e}")

st.write("### About")
st.write(
    "The Object Recognition System leverages deep learning models to classify images into predefined categories. "
    "The app uses two advanced models: CNN and MobileNetV2. After uploading an image, you can get predictions "
    "from the selected model and view detailed performance metrics including confusion matrix, precision-recall curve, ROC curve, training metrics, weighted averages of metrics, "
    "and dataset distribution. The CIFAR-10 dataset, which includes classes such as airplane, automobile, bird, cat, deer, "
    "dog, frog, horse, ship, and truck, is used for evaluation. Use the dropdowns to select the model and the type of "
    "visualization you want to see."
)