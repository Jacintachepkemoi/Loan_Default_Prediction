import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page configuration and header styling
st.set_page_config(page_title="AI Loan Approval Dashboard", layout="centered")
st.title("🏦 AI Loan Default Risk Predictor")
st.write("Input applicant details below to calculate the real-time default probability based on top model drivers.")

# 2. Load pipeline tools dynamically and inject true dataset means
@st.cache_resource
def load_pipeline_tools():
    # Production file linkages
    scaler = joblib.load("C:/Users/jacinta/Documents/My_Data_Project/loan_scaler.pkl")
    model = joblib.load(r"C:\Users\jacinta\Documents\My_Data_Project\champion_loan_model.pkl")
    
    # Extract exact column layout signatures directly from the scaler asset
    trained_columns = list(scaler.feature_names_in_)
    
    # MATHEMATICALLY PERFECT BASELINE MEANS: Sourced directly from your 255k borrower history
    baseline_means = {
        'Education_High School': 0.250259, 
        "Education_Master's": 0.248842, 
        'Education_PhD': 0.248826,
        'EmploymentType_Part-time': 0.251270, 
        'EmploymentType_Self-employed': 0.249488, 
        'EmploymentType_Unemployed': 0.249950,
        'MaritalStatus_Married': 0.334063, 
        'MaritalStatus_Single': 0.332927,
        'HasMortgage_Yes': 0.500014, 
        'HasCoSigner_Yes': 0.500108
    }
    return scaler, model, trained_columns, baseline_means

scaler, model, expected_columns, baseline_means = load_pipeline_tools()

# 3. Interactive user interface widgets mapping to top drivers
st.header("📊 Top Financial Risk Factors")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Applicant Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000, step=1000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
    months_employed = st.slider("Months Employed", min_value=0, max_value=120, value=24)

with col2:
    loan_term = st.slider("Loan Term (Months)", min_value=12, max_value=60, value=36) 
    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01) 
    interest_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=30.0, value=5.5, step=0.1) 
    num_credit_lines = st.slider("Number of Credit Lines", min_value=0, max_value=15, value=2)
    has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Home", "Education", "Business", "Other"])

# 4. Processing framework and localized prediction pipeline
if st.button("🚀 Calculate Default Risk", type="primary"):
    
    # Establish full 24-column empty schema template
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    # Seed silent categorical fields with true data population averages
    for col, mean_val in baseline_means.items():
        if col in input_df.columns:
            input_df[col] = mean_val
            
    # Inject active slider attributes into explicit feature keys
    input_df['Age'] = age
    input_df['Income'] = income
    input_df['LoanAmount'] = loan_amount
    input_df['CreditScore'] = credit_score
    input_df['MonthsEmployed'] = months_employed
    input_df['LoanTerm'] = loan_term
    input_df['DTIRatio'] = dti_ratio
    input_df['InterestRate'] = interest_rate
    input_df['NumCreditLines'] = num_credit_lines
    input_df['HasDependents_Yes'] = 1 if has_dependents == "Yes" else 0
    
    # Isolate and explicitly set loan intent configurations
    input_df['LoanPurpose_Home'] = 0
    input_df['LoanPurpose_Education'] = 0
    input_df['LoanPurpose_Business'] = 0
    input_df['LoanPurpose_Other'] = 0
    
    if loan_purpose == "Home": input_df['LoanPurpose_Home'] = 1
    elif loan_purpose == "Education": input_df['LoanPurpose_Education'] = 1
    elif loan_purpose == "Business": input_df['LoanPurpose_Business'] = 1
    elif loan_purpose == "Other": input_df['LoanPurpose_Other'] = 1

    try:
        # Core transformation and prediction mapping
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
