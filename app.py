import streamlit as st
import numpy as np
import joblib

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Approval Predictor")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
ApplicantIncome = st.number_input("Applicant Income")
CoapplicantIncome = st.number_input("Coapplicant Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Amount Term")
Credit_History = st.selectbox("Credit History", [0, 1])
Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Manual encoding
gender = 1 if Gender == "Male" else 0
married = 1 if Married == "Yes" else 0

dep = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}[Dependents]

education = 0 if Education == "Graduate" else 1
self_emp = 1 if Self_Employed == "Yes" else 0

property_area = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[Property_Area]

input_data = np.array([[
    gender,
    married,
    dep,
    education,
    self_emp,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    property_area
]])

input_data = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")