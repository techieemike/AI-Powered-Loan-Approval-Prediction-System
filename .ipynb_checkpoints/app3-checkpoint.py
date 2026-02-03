import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
from lime import lime_tabular
from streamlit_shap import st_shap
import streamlit.components.v1 as components

# 1. Load the model and training columns (crucial for feature names)
model_path = r"C:\Users\ABIMIC\Documents\Study\Projects\Machine Learning\AI Loan Approval Prediction Systems\model\loan_model_stacked.pkl"
model = joblib.load(model_path)
feature_names = model.feature_names_in_ # Get the exact names used in training

st.set_page_config(page_title="Loan Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction with Explainable AI")

# --- UI Sidebar for Inputs ---
with st.sidebar:
    st.header("Applicant Details")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Married", ['Yes', 'No'])
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
    credit_history = st.selectbox("Credit History", ['1.0 (Good)', '0.0 (Bad)'])
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
    applicant_income = st.number_input("Applicant Income", min_value=1)
    loan_amount = st.number_input("Loan Amount", min_value=1)
    loan_term = st.number_input("Loan Term", value=360)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)

# 2. Replicate your Feature Engineering Exactly
total_income = applicant_income + coapplicant_income
loan_to_income = loan_amount / total_income
# Note: Ensure the columns here match your get_dummies output exactly!
data = {
    'Credit_History': 1.0 if '1.0' in credit_history else 0.0,
    'Log_ApplicantIncome': np.log1p(applicant_income),
    'Log_LoanAmount': np.log1p(loan_amount),
    'Log_Total_Income': np.log1p(total_income),
    'Log_Loan_to_Income': np.log1p(loan_to_income),
    'Monthly_Installment': loan_amount / loan_term,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Married_Yes': 1 if married == 'Yes' else 0,
    'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
    'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
    'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
    'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
}

# Create DataFrame and ensure column order matches the model
X_input = pd.DataFrame([data]).reindex(columns=feature_names, fill_value=0)

# --- Prediction Logic ---
if st.button("Analyze Loan Application"):
    prob = model.predict_proba(X_input)[0, 1]
    
    # Use your custom threshold 0.31
    prediction = 1 if prob >= 0.31 else 0
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.success(f"### ✅ Approved (Confidence: {prob:.2%})")
        else:
            st.error(f"### ❌ Rejected (Confidence: {1-prob:.2%})")

    # --- SHAP Explanation ---
    st.subheader("🔍 Global Influence (SHAP Force Plot)")
    # We use a placeholder for background data or just a simplified explainer
    # For a live app, we often use a pre-computed explainer to save time
    explainer = shap.Explainer(model.predict_proba, X_input) # Simplified for single row
    shap_values = explainer(X_input)
    
    # Render SHAP
    st_shap(shap.plots.force(shap_values[0][:, 1]), height=200)

    # --- LIME Explanation ---
    st.subheader("🧪 Why this specific result? (LIME)")
    # Note: LIME usually needs a training set reference. 
    # For this demo, we initialize it simply, but in production, 
    # you'd load a small sample of X_train_bal.values
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.zeros((1, len(feature_names))), # Placeholder, use real sample for better results
        feature_names=feature_names,
        class_names=['Rejected', 'Approved'],
        mode='classification'
    )
    
    exp = lime_explainer.explain_instance(X_input.values[0], model.predict_proba, num_features=5)
    components.html(exp.as_html(), height=400, scrolling=True)