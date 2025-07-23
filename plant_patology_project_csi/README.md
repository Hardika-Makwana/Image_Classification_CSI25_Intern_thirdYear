# Plant Pathology Project CSI
This folder in the repository contains the image classification project assigned for the Plant Pathology task, which focuses on classifying leaf images into multiple disease categories using traditional machine learning techniques.
Project Description
This folder in the repository contains the image classification project assigned for the Plant Pathology task.
The aim is to classify leaf images into four categories — healthy, rust, scab, and multiple diseases — using traditional machine learning techniques instead of deep learning or transfer learning.

The project demonstrates how handcrafted features such as color histograms, Local Binary Patterns (LBP), and Haralick GLCM texture properties can be extracted from images to train classical ML models.

Three main models were implemented:
-Random Forest
-Gradient Boosting
-Support Vector Machine (SVM)
and additionally, XGBoost was tested to compare performance.

DATASET DESCRIPTION

Images of plant leaves with different conditions (healthy or infected with various diseases).
Each image has a unique image_id and a label.

>Labels are encoded as:
-healthy (516 images)
-rust (622 images)
-scab (592 images)
-multiple_diseases (91 images)
Images are stored in a local folder: images_plant_pathology/
Labels are linked via a CSV file: plant_pathology_project.csv

FEATURES EXTRACTED
Color Histograms:
8 bins each for Blue, Green, and Red channels (24 features).

Texture (LBP):
Local Binary Pattern histograms (9 features).

Texture (GLCM):
Haralick Gray-Level Co-occurrence Matrix properties- contrast and dissimilarity (2 features).

MODEL TRAINING AND EVALUATION
-Data was split into training and validation sets.
-Each model was trained using GridSearchCV to find optimal parameters.

>Evaluation included:
-Validation Accuracy
-Confusion Matrix
-Classification Report (Precision, Recall, F1-score)
-Models were also tested on random validation images to check prediction correctness visually.

TOOLS AND LIBRARIES
-Python
-OpenCV (image reading & processing)
-Scikit-image (LBP & GLCM texture)
-Scikit-learn (models, metrics, GridSearchCV)
-XGBoost (gradient boosting classifier)
-Matplotlib & Seaborn (visualization)

OUTCOME
-Gradient Boosting achieved the best accuracy (~61%).
-Random Forest and XGBoost performed comparably (~58–59%).
-SVM had slightly lower accuracy (~58%).
-Accuracy is lower than deep learning approaches because traditional ML models rely on manually engineered features, not end-to-end feature learning.

LIMITATIONS
-No use of test images: the dataset’s official test images had no ground truth labels and were meant for Kaggle submission only, so they were not used for final evaluation.
-Accuracy can be significantly improved using deep learning (CNNs) and transfer learning techniques.
-Handcrafted features can’t capture complex patterns as effectively as learned features from CNN filters.


