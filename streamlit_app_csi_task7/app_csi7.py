import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#  Load  trained model
with open('streamlit_app_csi_task7/model_csi7.pkl', 'rb') as f:
    model = pickle.load(f)

#  Load Iris dataset for visuals
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
species_names = iris.target_names

#  Streamlit page
st.title("CSI Task 7 - Iris Prediction App")

st.header("Enter flower measurements:")

# Sliders for input
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# Combine input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict when button is clicked
if st.button(' Predict'):
    prediction = model.predict(input_data)[0]
    prediction_name = species_names[prediction].capitalize()

    st.subheader(f" Predicted Species: **{prediction_name}**")

    prob = model.predict_proba(input_data)[0]
    prob_df = pd.DataFrame(prob, index=species_names, columns=['Probability'])
    st.subheader(" Prediction Probabilities")
    st.bar_chart(prob_df)

    st.subheader(" Your Input Data")
    input_df = pd.DataFrame(input_data, columns=iris.feature_names)
    st.table(input_df)

    st.subheader(" Sepal Length vs Sepal Width")
    fig, ax = plt.subplots()
    for i, species_name in enumerate(species_names):
        df = iris_df[iris_df['species'] == i]
        ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'], label=species_name)
    ax.scatter(sepal_length, sepal_width, color='black', marker='X', s=100, label='Your Input')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.legend()
    st.pyplot(fig)

