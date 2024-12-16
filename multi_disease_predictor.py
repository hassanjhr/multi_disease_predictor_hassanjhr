# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:40:15 2024

@author: Precision
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained models (relative paths)
diabetes_model = pickle.load(open('trained_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Diabetes Prediction Function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = diabetes_model.predict(input_data_as_numpy_array)
    return '🎉 The person is **not diabetic**.' if prediction[0] == 0 else '⚠️ The person is **diabetic**.'

# Heart Disease Prediction Function
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = heart_disease_model.predict(input_data_as_numpy_array)
    return '🎉 The person is **not at risk for heart disease**.' if prediction[0] == 0 else '⚠️ The person is **at risk for heart disease**.'

# Main Streamlit App
def main():
    # Page configuration
    st.set_page_config(page_title="Multi-Disease Prediction App", page_icon="💡", layout="centered")
    
    # Sidebar for Navigation
    st.sidebar.title("🔍 Disease Prediction")
    choice = st.sidebar.radio("Select a Disease:", ["Diabetes Prediction", "Heart Disease Prediction"])
    
    # Styling for Results
    st.markdown("""
        <style>
            .result-success {
                background-color: #D4EDDA;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.2rem;
                margin: 20px 0;
            }
            .result-danger {
                background-color: #F8D7DA;
                color: #721C24;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.2rem;
                margin: 20px 0;
                animation: blink 1.5s infinite;
            }
            @keyframes blink {
                0% {opacity: 1;}
                50% {opacity: 0.5;}
                100% {opacity: 1;}
            }
        </style>
    """, unsafe_allow_html=True)

    # Diabetes Prediction
    if choice == "Diabetes Prediction":
        st.title("🩺 Diabetes Prediction Web App")
        
        # Input features for Diabetes Prediction
        st.write("### Input Details")
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
        Glucose = st.number_input('Glucose Level', min_value=0)
        BloodPressure = st.number_input('Blood Pressure value', min_value=0)
        SkinThickness = st.number_input('Skin Thickness value', min_value=0)
        Insulin = st.number_input('Insulin Level', min_value=0)
        BMI = st.number_input('BMI value', min_value=0.0, format="%.2f")
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, format="%.3f")
        Age = st.number_input('Age of the Person', min_value=0, step=1)

        if st.button("🧪 Predict Diabetes"):
            result = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            
            if "not diabetic" in result:
                st.markdown(f'<div class="result-success">{result}</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="result-danger">{result}</div>', unsafe_allow_html=True)
                st.error("⚠️ **Warning: High risk detected! Please consult a doctor immediately.**")

    # Heart Disease Prediction
    elif choice == "Heart Disease Prediction":
        st.title("💓 Heart Disease Prediction Web App")
        
        # Input features for Heart Disease Prediction
        st.write("### Input Details")
        age = st.number_input('Age', min_value=0, step=1)
        sex = st.number_input('Sex (0 = Female, 1 = Male)', min_value=0, max_value=1)
        cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3)
        trestbps = st.number_input('Resting Blood Pressure', min_value=0)
        chol = st.number_input('Serum Cholesterol in mg/dL', min_value=0)
        fbs = st.number_input('Fasting Blood Sugar (>120 mg/dL: 1, otherwise: 0)', min_value=0, max_value=1)
        restecg = st.number_input('Resting Electrocardiographic Results (0-5)', min_value=0, max_value=5)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)
        exang = st.number_input('Exercise Induced Angina (0 = No, 1 = Yes)', min_value=0, max_value=1)
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, format="%.2f")
        slope = st.number_input('Slope of the Peak Exercise ST Segment (0-5)', min_value=0, max_value=5)
        ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy (0-5)', min_value=0, max_value=5)
        thal = st.number_input('Thalassemia (0-5)', min_value=0, max_value=5)

        if st.button("💓 Predict Heart Disease"):
            result = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
            
            if "not at risk" in result:
                st.markdown(f'<div class="result-success">{result}</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="result-danger">{result}</div>', unsafe_allow_html=True)
                st.error("⚠️ **Warning: High risk detected! Please consult a doctor immediately.**")

if __name__ == "__main__":
    main()
