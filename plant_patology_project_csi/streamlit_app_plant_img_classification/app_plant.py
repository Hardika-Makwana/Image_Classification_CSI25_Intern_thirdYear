import joblib
import os

current_dir = os.path.dirname(__file__)   # This is streamlit-app/
model_path = os.path.join(current_dir, "gbm_model_plant_csi.pkl")

gbm_model = joblib.load(model_path)
print("Model loaded successfully!")

