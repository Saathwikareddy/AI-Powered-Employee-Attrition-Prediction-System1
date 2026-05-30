import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📞 Customer Churn Prediction")
st.write("Enter customer details below.")

# Inputs
gender = st.selectbox("Gender", [0, 1])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", [0, 1])
Dependents = st.selectbox("Dependents", [0, 1])

tenure = st.number_input("Tenure", min_value=0)

PhoneService = st.selectbox("Phone Service", [0, 1])
MultipleLines = st.selectbox("Multiple Lines", [0, 1])

InternetService = st.number_input("Internet Service (Encoded Value)", value=0)
OnlineSecurity = st.number_input("Online Security (Encoded Value)", value=0)
OnlineBackup = st.number_input("Online Backup (Encoded Value)", value=0)
DeviceProtection = st.number_input("Device Protection (Encoded Value)", value=0)
TechSupport = st.number_input("Tech Support (Encoded Value)", value=0)
StreamingTV = st.number_input("Streaming TV (Encoded Value)", value=0)
StreamingMovies = st.number_input("Streaming Movies (Encoded Value)", value=0)

Contract = st.number_input("Contract (Encoded Value)", value=0)

PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])

PaymentMethod = st.number_input("Payment Method (Encoded Value)", value=0)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)

TotalCharges = st.number_input("Total Charges", min_value=0.0)

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

    probability = model.predict_proba(input_scaled)[0][1]

    if probability > 0.5:
        st.error(
            f"⚠ Customer is likely to CHURN\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY\n\nProbability: {(1-probability):.2%}"
        )

    st.write("Churn Probability:", round(probability, 4))
