
import streamlit as st
import numpy as np
import pickle

# Path to saved model
model_path = r"C:\Users\ABIMIC\Documents\Projects\AI\Machine Learning\AI Loan Approval Prediction 1\Loan_Approval_Prediction-main\model\loan_approval_model.pkl"

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")

# Input form
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
credit_history = st.selectbox("Credit History", ['1.0 (Good)', '0.0 (Bad)'])
property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
applicant_income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)

# Log-transform numeric values
log_applicant_income = np.log1p(applicant_income)
log_loan_amount = np.log1p(loan_amount)
log_loan_term = np.log1p(loan_term)
log_total_income = np.log1p(applicant_income + coapplicant_income)

# Encode categorical variables
gender = 1 if gender == 'Male' else 0
married = 1 if married == 'Yes' else 0
dependents = 3 if dependents == '3+' else int(dependents)
education = 0 if education == 'Graduate' else 1
self_employed = 1 if self_employed == 'Yes' else 0
credit_history = 1.0 if '1.0' in credit_history else 0.0
property_area = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}[property_area]

# Feature vector
X_input = np.array([[gender, married, dependents, education, self_employed, credit_history,
                     property_area, log_applicant_income, log_loan_amount,
                     log_loan_term, log_total_income]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(X_input)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.success(result)
