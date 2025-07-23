import streamlit as st
import joblib
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# === Load your model ===
current_dir = os.path.dirname(__file__)  # This script’s folder
model_path = os.path.join(current_dir, "gbm_model_plant_csi.pkl")
gbm_model = joblib.load(model_path)

st.title(" Plant Disease Classifier (Gradient Boosting)")

st.write(
    """
    Upload a leaf image and this app will classify it as **healthy**, **rust**, **scab**, or **multiple diseases**
    using a **Gradient Boosting** model trained on handcrafted features.
    """
)

# === Upload image ===
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # === Extract features ===
    def extract_features(img):
        img = cv2.resize(img, (128, 128))

        chans = cv2.split(img)
        hist_features = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]

        features = np.concatenate([hist_features, lbp_hist, [contrast, dissimilarity]])
        return features.reshape(1, -1)

    features = extract_features(image)

    # === Make prediction ===
    pred_encoded = gbm_model.predict(features)[0]

    # Your correct label map:
    label_map = {
        0: "healthy",
        1: "multiple_diseases",
        2: "rust",
        3: "scab"
    }
    pred_label = label_map.get(pred_encoded, "Unknown")

    st.success(f"**Prediction:** {pred_label}")



