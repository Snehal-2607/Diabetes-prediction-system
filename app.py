# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:47:01 2025

@author: HP
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('diabetes_prediction_model.pkl')
except FileNotFoundError:
    st.error("Error: 'diabetes_prediction_model.pkl' not found. Please ensure the model is saved in the same directory.")
    st.stop()

# Define the mappings for categorical features (from your original notebook)
gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
smoking_mapping = {
    'never': 0,
    'No Info': 1,
    'current': 2,
    'former': 3,
    'ever': 4,
    'not current': 5
}

# --- Streamlit UI ---
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction System")
st.markdown("Enter the patient's details to predict the likelihood of diabetes.")

st.header("Patient Information Input")

# Create input widgets for each feature
col1, col2 = st.columns(2)

with col1:
    gender_input = st.selectbox("Gender", options=list(gender_mapping.keys()))
    age = st.slider("Age", 0.0, 100.0, 30.0)
    hypertension = st.radio("Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heart_disease = st.radio("Heart Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    smoking_history_input = st.selectbox("Smoking History", options=list(smoking_mapping.keys()))
    bmi = st.number_input("BMI (Body Mass Index)", value=25.0, min_value=10.0, max_value=60.0)
    hba1c_level = st.number_input("HbA1c Level", value=5.7, min_value=3.0, max_value=10.0, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", value=120, min_value=50, max_value=300, step=1)

# Prediction button
if st.button("Predict Diabetes"):
    # Preprocess the input data
    gender_encoded = gender_mapping[gender_input]
    smoking_history_encoded = smoking_mapping[smoking_history_input]

    input_data = pd.DataFrame({
        'gender': [gender_encoded],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history_encoded],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"The model predicts **DIABETES** (Probability: {prediction_proba[0][1]*100:.2f}%)")
        st.image("https://i.imgur.com/G53wI9P.png", caption="Diabetes Detected", width=200) # Example image for positive prediction
    else:
        st.success(f"The model predicts **NO DIABETES** (Probability: {prediction_proba[0][0]*100:.2f}%)")
        st.image("https://i.imgur.com/2P9zW6X.png", caption="No Diabetes Detected", width=200) # Example image for negative prediction

    st.write("---")
    st.info("Disclaimer: This prediction is based on a machine learning model and should not be considered a substitute for professional medical advice.")