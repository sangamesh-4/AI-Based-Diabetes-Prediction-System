import streamlit as st
import pandas as pd
import pickle

# -------- PAGE SETTINGS --------
st.set_page_config(
    page_title="AI Diabetes Prediction",
    page_icon="🧠",
    layout="wide"
)

# -------- CUSTOM CSS --------
st.markdown("""
<style>

.title{
    font-size:85px;
    font-weight:800;
    color:#2E86C1;
    text-align:center;
}

.subtitle{
    font-size:22px;
    color:gray;
    text-align:center;
}

.info-box{
    background-color:#eef5ff;
    padding:25px;
    border-radius:10px;
    margin-bottom:20px;
}

.result-high{
    background-color:#f8d7da;
    padding:25px;
    border-radius:10px;
    font-size:24px;
    color:#721c24;
    text-align:center;
}

.result-low{
    background-color:#d4edda;
    padding:25px;
    border-radius:10px;
    font-size:24px;
    color:#155724;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -------- TITLE --------
st.markdown('<p class="title">🧠 AI-Based Diabetes Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient health parameters to predict diabetes risk using Machine Learning</p>', unsafe_allow_html=True)

# -------- LOAD MODEL --------
loaded_model = pickle.load(open("diabetes_model.pkl", "rb"))

# -------- SIDEBAR --------
st.sidebar.header("🩺 Patient Details")

preg = st.sidebar.slider("Pregnancies", 0, 17, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 130, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 21, 81, 30)

# -------- INPUT DATA --------
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

# -------- RIGHT SIDE CONTENT --------

st.markdown(
"""
<div class="info-box">

### 📌 How to Use This App

1️⃣ Enter patient health details in the **left sidebar**.  
2️⃣ Adjust the sliders according to patient medical values.  
3️⃣ Click **Predict Diabetes** to analyze the risk.

This system uses a **Logistic Regression Machine Learning model** trained on medical data to estimate diabetes probability.

</div>
""",
unsafe_allow_html=True
)

# -------- HEALTH FACTORS --------
st.markdown("### 🔍 Important Health Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
**Glucose Level**

High glucose levels are the strongest indicator of diabetes risk.
""")

with col2:
    st.markdown("""
**BMI (Body Mass Index)**

Higher BMI often indicates obesity which increases diabetes risk.
""")

with col3:
    st.markdown("""
**Age Factor**

Diabetes risk increases with age, especially after 40 years.
""")

# -------- PREDICTION --------

if st.sidebar.button("🔍 Predict Diabetes"):

    st.markdown("### 📊 Entered Patient Data")
    st.write(features)

    prediction = loaded_model.predict(features)

    st.markdown("### 🧾 Prediction Result")

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-high">⚠️ High Risk: Patient is likely to have Diabetes</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-low">✅ Low Risk: Patient is unlikely to have Diabetes</div>',
            unsafe_allow_html=True
        )
