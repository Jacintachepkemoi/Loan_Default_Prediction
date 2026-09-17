import joblib
import pandas as pd

# 1. Load the data spreadsheet
print("--- Loading Dataset ---")
df = pd.read_csv("C:/Users/jacinta/Documents/My_Data_Project/Loan_default.csv")
print(f"Successfully loaded {len(df)} loan rows!")
print("Columns in data:", list(df.columns[:5]), "... and more.")

# 2. Load the preprocessing scaler
scaler = joblib.load("C:/Users/jacinta/Documents/My_Data_Project/loan_scaler.pkl")
print("\n--- Scaler Successfully Loaded! ---")

# 3. Load the champion loan model
model = joblib.load(r"C:\Users\jacinta\Documents\My_Data_Project\champion_loan_model.pkl")
print("--- Model Successfully Loaded! ---")

# 4. Preview the first few rows of data
print("\n--- First 5 Rows of Loan Data ---")
print(df.head())
