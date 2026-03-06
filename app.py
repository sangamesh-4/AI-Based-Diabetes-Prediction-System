import streamlit as st
import pandas as pd
import pickle

# Title
st.title("AI-Based Diabetes Prediction System")
st.write("Enter patient details and click submit to predict diabetes risk.")

# Load trained model
loaded_model = pickle.load(open("diabetes_model.pkl", "rb"))

# Sidebar inputs
st.sidebar.title("Patient Details")

preg = st.sidebar.slider("Pregnancies", 0, 17, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 130, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 21, 81, 30)

# Create dataframe from input
features = pd.DataFrame({
    "Pregnancies":[preg],
    "Glucose":[glucose],
    "BloodPressure":[bp],
    "SkinThickness":[skin],
    "Insulin":[insulin],
    "BMI":[bmi],
    "DiabetesPedigreeFunction":[dpf],
    "Age":[age]
})

# Submit button
if st.sidebar.button("Submit"):

    st.subheader("Entered Patient Data")
    st.write(features)

    prediction = loaded_model.predict(features)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk: The patient is likely to have Diabetes")
    else:
        st.success("✅ Low Risk: The patient is not likely to have Diabetes")