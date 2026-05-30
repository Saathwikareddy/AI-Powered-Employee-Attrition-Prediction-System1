import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Telco Customer Churn Prediction")

st.title("📞 Telco Customer Churn Prediction")

# Inputs
gender = st.selectbox("Gender", [0, 1])

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

Partner = st.selectbox("Partner", [0, 1])

Dependents = st.selectbox("Dependents", [0, 1])

tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)

PhoneService = st.selectbox("Phone Service", [0, 1])

MultipleLines = st.selectbox("Multiple Lines", [0, 1, 2])

InternetService = st.selectbox(
    "Internet Service",
    [0, 1, 2]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    [0, 1, 2]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    [0, 1, 2]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    [0, 1, 2]
)

TechSupport = st.selectbox(
    "Tech Support",
    [0, 1, 2]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    [0, 1, 2]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    [0, 1, 2]
)

Contract = st.selectbox(
    "Contract",
    [0, 1, 2]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    [0, 1]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [0, 1, 2, 3]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

if st.button("Predict Churn"):

    input_data = np.array([[
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    if probability > 0.5:
        st.error(
            f"⚠ Customer is likely to CHURN\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY\n\nProbability: {(1 - probability):.2%}"
        )

    st.write("Churn Probability:", round(float(probability), 4))
