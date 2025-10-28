import os
import joblib
import warnings
import logging
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Suppress sklearn InconsistentVersionWarning
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Suppress Streamlit ScriptRunContext warnings
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

class _StreamlitScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if record.name.startswith("streamlit.runtime.scriptrunner_utils.script_run_context") and "missing ScriptRunContext" in msg:
            return False
        return True

logging.getLogger().addFilter(_StreamlitScriptRunContextFilter())

# Load models
model_path = '.'
try:
    loaded_knn_model = joblib.load(os.path.join(model_path, 'knn_model.pkl'))
    loaded_nb_model = joblib.load(os.path.join(model_path, 'nb_model.pkl'))
    le = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))  # Load fitted LabelEncoder
    print("Models and label encoder loaded successfully!")
except FileNotFoundError:
    print("Error: Required files not found.")
    loaded_knn_model = None
    loaded_nb_model = None
    le = None

# Streamlit UI
st.title("Animal Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image_placeholder = st.empty()
classify_button = st.button("Classify Image")
result_placeholder = st.empty()

# Feature extraction
def extract_histogram_gui(image, size=(64, 64)):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, size)
        hist = cv2.calcHist([resized_image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, None).flatten()
        return hist
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Prediction logic
def predict_animal_gui(uploaded_file, knn_model, nb_model, class_names):
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Could not read the uploaded image.")
                return None, None

            image_placeholder.image(image, caption='Uploaded Image.', use_container_width=True)
            features = extract_histogram_gui(image)

            if features is None:
                return None, None

            features = features.reshape(1, -1)
            knn_prediction_index = knn_model.predict(features)[0]
            nb_prediction_index = nb_model.predict(features)[0]

            knn_predicted_class = class_names[knn_prediction_index]
            nb_predicted_class = class_names[nb_prediction_index]

            return knn_predicted_class, nb_predicted_class

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    else:
        return None, None

# Trigger classification
if classify_button and uploaded_file is not None:
    if le is not None:
        class_names = le.classes_
    elif loaded_knn_model is not None and hasattr(loaded_knn_model, "classes_"):
        class_names = loaded_knn_model.classes_
    elif loaded_nb_model is not None and hasattr(loaded_nb_model, "classes_"):
        class_names = loaded_nb_model.classes_
    else:
        class_names = None

    if class_names is None:
        result_placeholder.error("Class names not found. Please ensure LabelEncoder or model classes are available.")
    elif loaded_knn_model is None or loaded_nb_model is None:
        result_placeholder.error("Models not loaded. Ensure 'knn_model.pkl' and 'nb_model.pkl' are available.")
    else:
        knn_prediction, nb_prediction = predict_animal_gui(uploaded_file, loaded_knn_model, loaded_nb_model, class_names)
        if knn_prediction is not None:
            result_placeholder.success(f"KNN Predicted Animal: {knn_prediction}")
            result_placeholder.success(f"Naive Bayes Predicted Animal: {nb_prediction}")
elif classify_button and uploaded_file is None:
    result_placeholder.warning("Please upload an image before classifying.")
