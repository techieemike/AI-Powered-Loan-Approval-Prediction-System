import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
from lime import lime_tabular
from streamlit_shap import st_shap
import streamlit.components.v1 as components

# --- 1. Load Model & Setup ---
# UPDATED: Relative path for local and cloud compatibility
model_path = "loan_model_stacked.pkl"

@st.cache_resource
def load_model():
    try:
        loaded_model = joblib.load(model_path)
        return loaded_model
    except FileNotFoundError:
        st.error(f"🚨 Model file '{model_path}' not found! Ensure it's in the root folder.")
        st.stop()

model = load_model()
feature_names = model.feature_names_in_

st.set_page_config(page_title="Loan AI Predictor", layout="wide")

# --- 2. Sidebar UI ---
st.sidebar.header("📋 Applicant Information")
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
married = st.sidebar.selectbox("Married", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])
credit_history = st.sidebar.selectbox("Credit History", ['1.0 (Good)', '0.0 (Bad)'])
property_area = st.sidebar.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

st.sidebar.subheader("💰 Financials")
applicant_income = st.sidebar.number_input("Applicant Income ($)", min_value=1, value=5000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income ($)", min_value=0, value=0)
loan_amount = st.sidebar.number_input("Loan Amount (In Thousands)", min_value=1, value=150)
loan_term = st.sidebar.number_input("Loan Term (Months)", min_value=1, value=360)

# --- 3. Feature Engineering ---
total_income = applicant_income + coapplicant_income
loan_to_income = loan_amount / total_income if total_income > 0 else 0
monthly_installment = loan_amount / loan_term if loan_term > 0 else 0

# Construct data dictionary for model consumption
data = {
    'Credit_History': 1.0 if '1.0' in credit_history else 0.0,
    'Log_ApplicantIncome': np.log1p(applicant_income),
    'Log_LoanAmount': np.log1p(loan_amount),
    'Log_Total_Income': np.log1p(total_income),
    'Log_Loan_to_Income': np.log1p(loan_to_income),
    'Monthly_Installment': monthly_installment,
    'Log_Loan_Term': np.log1p(loan_term),
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Married_Yes': 1 if married == 'Yes' else 0,
    'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
    'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
    'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
    'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
    'Dependents_1': 1 if dependents == '1' else 0,
    'Dependents_2': 1 if dependents == '2' else 0,
    'Dependents_3+': 1 if dependents == '3+' else 0
}

# Align columns with model expectations
X_input = pd.DataFrame([data]).reindex(columns=feature_names, fill_value=0)

# --- 4. Main Display ---
st.title("🏦 AI Loan Approval Prediction System")
st.markdown("Developed by **Abikale Michael Raymond**. This system utilizes a **Stacked Ensemble Model** for high-precision forecasting.")

if st.button("Analyze Application", use_container_width=True):
    # Prediction logic
    prob = model.predict_proba(X_input)[0, 1]
    prediction = 1 if prob >= 0.31 else 0  # Optimized Threshold
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.success("### Decision: ✅ APPROVED")
        else:
            st.error("### Decision: ❌ REJECTED")
        st.write(f"**Confidence Score:** {prob:.2%}")
        st.progress(prob)

    with res_col2:
        st.info("💡 **Model Logic:** This decision is based on a probability threshold of **0.31**, optimized to balance risk and approval rates.")

    # --- 5. Explainable AI (XAI) Section ---
    st.write("---")
    tab1, tab2 = st.tabs(["🔍 Case-Specific Explanation (LIME)", "📊 Overall Feature Impact (SHAP)"])

    with tab1:
        st.subheader("Decision Breakdown (LIME)")
        # Note: Using zeros as baseline for app speed; ideally replaced by small training sample
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.zeros((1, len(feature_names))), 
            feature_names=list(feature_names),
            class_names=['Rejected', 'Approved'],
            mode='classification'
        )
        exp = lime_explainer.explain_instance(X_input.values[0], model.predict_proba, num_features=10)
        components.html(exp.as_html(), height=400, scrolling=True)

    with tab2:
        st.subheader("SHAP Contribution Plot")
        explainer = shap.Explainer(model.predict_proba, X_input)
        shap_values = explainer(X_input)
        # Focus on probability of Approval (Class 1)
        st_shap(shap.plots.force(shap_values[0][:, 1]), height=200)