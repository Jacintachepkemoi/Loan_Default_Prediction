import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page settings and title
st.set_page_config(page_title="AI Loan Approval Dashboard", layout="centered")
st.title("🏦 AI Loan Default Risk Predictor")
st.write("Input applicant details below to calculate the real-time default probability based on top model drivers.")

# 2. Load models and learn the exact column structure from your CSV
@st.cache_resource
def load_pipeline_tools():
    scaler = joblib.load("C:/Users/jacinta/Documents/My_Data_Project/loan_scaler.pkl")
    model = joblib.load(r"C:\Users\jacinta\Documents\My_Data_Project\champion_loan_model.pkl")
    
    # Load template column structure
    df_template = pd.read_csv("C:/Users/jacinta/Documents/My_Data_Project/Loan_default.csv", nrows=100)
    X_template = df_template.drop(columns=["LoanID", "Default"])
    X_encoded_template = pd.get_dummies(X_template, drop_first=True)
    
    return scaler, model, X_encoded_template.columns

scaler, model, expected_columns = load_pipeline_tools()

# 3. Create the Interactive User Inputs based on true feature importances
st.header("📊 Top Financial Risk Factors")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Applicant Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000, step=1000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
    months_employed = st.slider("Months Employed", min_value=0, max_value=120, value=24)

with col2:
    loan_term = st.slider("Loan Term (Months)", min_value=12, max_value=60, value=36) # Rank 1 (325)
    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01) # Rank 2 (279)
    interest_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=30.0, value=5.5, step=0.1) # Rank 3 (261)
    num_credit_lines = st.slider("Number of Credit Lines", min_value=0, max_value=15, value=2)
    has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

# 4. Process Inputs and Predict when the button is clicked
if st.button("🚀 Calculate Default Risk", type="primary"):
    
    # Create a blank row matching template
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    # Inject values dynamically
    if 'Age' in input_df.columns: input_df['Age'] = age
    if 'Income' in input_df.columns: input_df['Income'] = income
    if 'LoanAmount' in input_df.columns: input_df['LoanAmount'] = loan_amount
    if 'CreditScore' in input_df.columns: input_df['CreditScore'] = credit_score
    if 'MonthsEmployed' in input_df.columns: input_df['MonthsEmployed'] = months_employed
    if 'LoanTerm' in input_df.columns: input_df['LoanTerm'] = loan_term
    if 'DTIRatio' in input_df.columns: input_df['DTIRatio'] = dti_ratio
    if 'InterestRate' in input_df.columns: input_df['InterestRate'] = interest_rate
    if 'NumCreditLines' in input_df.columns: input_df['NumCreditLines'] = num_credit_lines
    if 'HasDependents_Yes' in input_df.columns: input_df['HasDependents_Yes'] = 1 if has_dependents == "Yes" else 0
    
    try:
        scaled_data = scaler.transform(input_df)
        probability = model.predict_proba(scaled_data)[0][1] * 100 
        
        st.subheader("🔮 Model Analysis Result")
        if probability > 50:
            st.error(f"⚠️ High Risk Applicant! Probability of Default: {probability:.2f}%")
        else:
            st.success(f"✅ Approved / Low Risk! Probability of Default: {probability:.2f}%")
            
    except Exception as e:
        st.error("An error occurred during scaling or prediction.")
        st.write("Error details:", e)
