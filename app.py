import streamlit as st
import pickle
import numpy as np
import joblib

# Load the trained Linear Regression model
# Note: Ensure these files exist in your Colab file sidebar
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's clinical parameters to estimate heart disease risk.")

# Creating two columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True (1)" if x == 1 else "False (0)")
    restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0-3)", options=[0, 1, 2, 3])

# Prepare the input array
# Important: The order must exactly match the order of columns in the dataframe used for training
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict Risk Score"):
    prediction = model.predict(features_scaled)
    
    # Since Linear Regression returns a continuous value:
    score = prediction[0]
    
    st.subheader(f"Risk Score: {score:.2f}")
    
    # Using 0.5 as a threshold for classification logic
    if score >= 0.5:
        st.error("⚠️ Prediction: Presence of Heart Disease (High Risk)")
    else:
        st.success("✅ Prediction: Absence of Heart Disease (Low Risk)")

st.info("Note: This model uses Linear Regression to calculate a risk probability score.")
