import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_model.pkl")

# Load dataset metadata
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

st.title("Iris Flower Classification 🌸")

st.write("### Input Flower Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.2)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Create input dataframe
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

# Predict class
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    predicted_class = target_names[prediction]
    
    st.success(f"The predicted iris species is: **{predicted_class}** 🌿")
