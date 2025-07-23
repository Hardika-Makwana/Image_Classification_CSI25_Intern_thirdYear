# 🌿 Plant Pathology Project — CSI Internship

This folder in the repository contains the **image classification project** assigned for the Plant Pathology task, which focuses on classifying leaf images into multiple disease categories using **traditional machine learning techniques** (not deep learning or transfer learning).

---

## 🎯 **Project Objective**

To classify leaf images into **four categories** — **Healthy**, **Rust**, **Scab**, and **Multiple Diseases** — using **handcrafted features** and traditional ML models.

---

## 🗂️ **Dataset Description**

- Images of plant leaves with different conditions (healthy or infected).
- Each image has a **unique `image_id`** and a **label**.
- **Classes:**
  - Healthy (516 images)
  - Rust (622 images)
  - Scab (592 images)
  - Multiple Diseases (91 images)
- Images are stored in: `images_plant_pathology/`
- Labels are linked via: `plant_pathology_project.csv`

---

## 🔍 **Features Extracted**

- 📊 **Color Histograms:**  
  - 8 bins each for **Blue**, **Green**, **Red** channels → 24 features total.
- 🌀 **Texture (LBP):**  
  - Local Binary Pattern histograms → 9 features.
- 🧩 **Texture (GLCM):**  
  - Haralick Gray-Level Co-occurrence Matrix → **Contrast** & **Dissimilarity** (2 features).

---

## ⚙️ **Models Used**

- 🌲 **Random Forest**
- 🎯 **Gradient Boosting**
- 🧩 **Support Vector Machine (SVM)**
- ⚡ **XGBoost** (for comparison)

---

## 🧪 **Model Training & Evaluation**

- Data split into **training** and **validation** sets.
- **GridSearchCV** used for tuning hyperparameters.
- **Metrics:**  
  - Validation Accuracy  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-Score)
- Models also tested on **random validation images** to visually confirm predictions.

---

## 📊 **Outcome**

- ✅ **Gradient Boosting** achieved **~61% accuracy**.
- ✅ **Random Forest & XGBoost**: **~58–59% accuracy**.
- ✅ **SVM**: ~58% accuracy.
- Note: Accuracy is lower than deep learning as **traditional ML uses manual features**.

---

## ⚠️ **Limitations**

- 📌 **No use of test images:** Official test images had no ground truth labels (Kaggle-only).
- 📌 Accuracy can be significantly improved with **deep learning (CNNs)** and **transfer learning**.
- 📌 Traditionsl ML and Handcrafted features have limitations in capturing complex patterns.

---

## 🚀 **Deployment**

The best performing **Gradient Boosting model** is deployed using **Streamlit**:

👉 [**View the Live App Here**](https://imageclassificationcsi25internthirdyear-o8tovuycmczd8tzrwvjpu6.streamlit.app/)

---

## 🛠️ **Tools & Libraries**

- **Python**
- **OpenCV** (image reading & processing)
- **Scikit-image** (LBP & GLCM texture)
- **Scikit-learn** (models, GridSearchCV, metrics)
- **XGBoost**
- **Matplotlib & Seaborn** (visualization)
- **Streamlit** (deployment)

---

✨ *This project demonstrates how classic ML pipelines can be built for image classification with handcrafted features in a resource-constrained setting.*  



