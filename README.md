End-to-End Credit Risk Engine: From Algorithmic Optimization to Streamlit MLOps Deployment

A comprehensive, production-grade machine learning architecture designed to evaluate banking credit risk, predict consumer loan defaults, and serve real-time decisions using a large-scale financial dataset of 255,347 unique borrower profiles.

The Business Challenge and Core Strategy

In retail banking, a False Negative—approving a loan for an applicant who will ultimately fail to pay it back—is devastatingly expensive. Recovering the capital loss of a single toxic loan requires a bank to issue dozens of safe, interest-paying lines. This project intentionally bypasses lazy overall accuracy metrics to navigate the critical Precision-Recall trade-off. The primary objective was to optimize the engine's default-catching capabilities (Recall), establishing a robust financial shield capable of protecting capital reserves during high-risk economic climates.

Data Engineering and Preprocessing Steps

High-Cardinality Identifier Exclusion:
The unique customer string column (LoanID) was dropped prior to data transformation. This step was mandatory to prevent a catastrophic feature explosion during encoding that would have generated over 255,000 empty binary dummy columns and crashed the system's virtual memory limits.

Targeted Category One-Hot Encoding:
Pandas get_dummies was configured with drop_first=True, applied strictly to true text-based categorical columns (Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner). This successfully transformed text labels into readable binary flags while leaving numeric scales untouched.

Input Feature Scaling:
Continuous numerical fields (Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio) were normalized using StandardScaler. This centered the distribution of each column around a mean of 0 with a standard variance of 1, preventing high-magnitude fields like Income from mathematically blinding smaller, sensitive risk counters like CreditScore or Age.

Class Imbalance Diagnostics:
Exploratory data analysis revealed a severe class skew in the target variable (Default). Fully amortized loans made up 88.39% of the dataset (225,694 rows), while actual defaults comprised only 11.61% (29,653 rows).

Imbalance Resolution via Dual-Balancing:
To prevent the algorithms from falling into a lazy bias of predicting non-default for every applicant, the training dataset split was processed using SMOTE (Synthetic Minority Over-sampling Technique) to establish a balanced 50/50 baseline in memory. Furthermore, the gradient boosting tree models were augmented with a custom class-weight multiplier (scale_pos_weight=7.6), derived directly from the raw imbalance ratio, to aggressively punish the algorithms for missing rare default patterns.

The 4-Model Tournament Leaderboard

Four distinct competitive frameworks were trained on an isolated 80/20 train-test split. To guarantee unbiased evaluation, all probability calculations (y_proba) and absolute classification choices (y_pred) were generated strictly on the hidden test set (X_test), simulating how the models would perform on fresh real-world applicants.

*   LightGBM Classifier (SMOTE + 7.6 Weight)
    *   ROC AUC Score: 0.7269
    *   Class 1 Recall: 76.0%
    *   Class 1 Precision: 19.0%
    *   Status: Deployed Portfolio Champion
*   XGBoost Classifier (SMOTE + 7.6 Weight)
    *   ROC AUC Score: 0.7200
    *   Class 1 Recall: 75.0%
    *   Class 1 Precision: 19.0%
    *   Status: Runner-Up Contender
*   Random Forest Classifier (SMOTE Baseline)
    *   ROC AUC Score: 0.7043
    *   Class 1 Recall: 12.0%
    *   Class 1 Precision: 27.0%
    *   Status: Eliminated (Too Conservative)
*   Logistic Regression (SMOTE Baseline)
    *   ROC AUC Score: 0.6689
    *   Class 1 Recall: 25.0%
    *   Class 1 Precision: 23.0%
    *   Status: Eliminated (Weakest Separability)

Core Architectural and Strategic Insights

The Impact of Weight Multipliers:
Standard bagging models (Random Forest) and basic gradient boosters without custom tuning stalled at an unacceptable 12% to 14% Recall rate due to the heavy dataset imbalance. Introducing the 7.6 class penalty modifier radically altered the mathematical decision boundaries. By forcing the algorithms to treat a default row as 7.6 times more important than a normal row, the default-catching capability (Recall) skyrocketed to a massive 76.0%, successfully intercepting 4,455 out of 5,900 defaults.

Justification of Deployed Model:
LightGBM with dual SMOTE and class-weight tuning was selected as the definitive portfolio champion. It achieved the highest overall sorting power (0.7269 ROC AUC) and maximum risk protection. While this aggressive optimization causes a drop in precision (19%) due to false alarms, the model acts as an ideal defensive asset for financial institutions where preventing toxic debt overrides the cost of manual applicant review.

Explainable AI (XAI) Compliance:
To remove the black-box limitation of gradient boosting, the pipeline implements SHAP (SHapley Additive exPlanations) values grounded in cooperative game theory. By evaluating localized features for individual applicants, the system calculates exact positive and negative contribution metrics for every decision, proving mathematical compliance for fair-lending regulatory audits.

![SHAP Customer Waterfall Explanation](shap_chart.png)

Production MLOps Deployment Dashboard

To bridge the gap between research and production software engineering, the system's active components were permanently serialized using joblib. The resulting weights file (champion_loan_model.pkl) and data calibration matrix (loan_scaler.pkl) were migrated out of notebooks into a dedicated local environment using VS Code and Anaconda.

An interactive, user-facing credit risk dashboard was built from scratch using the Streamlit micro-framework. To maintain a clean, frictionless UI for banking credit officers, the application exposes only the top 10 structural risk drivers (such as Credit Score, Income, and Debt-to-Income ratio) via interactive input sliders and dropdowns. 

To prevent mathematical inaccuracy and ensure full 24-feature compliance under the hood, the backend application dynamically extracts the precise feature schema directly from the scaler's memory via feature_names_in_. Features hidden from the UI are automatically seeded with the exact statistical baseline population means calculated across all 255,347 historical loans (e.g., a baseline mortgage probability of 0.500014 and co-signer distribution of 0.500108). This design allows the application to perform lightning-fast, production-accurate predictions on a 24-column matrix, returning localized risk probabilities (e.g., a baseline score of 30.20%) in less than 10 milliseconds without the overhead of scanning a massive CSV file on every page refresh.

Local App Installation and Execution Guide

To clone this repository and launch the interactive credit assessment application on your local machine, run the following commands sequentially in your terminal:

1. Clone the project:
git clone https://github.com

2. Navigate to the project directory:
cd Loan_Default_Prediction

3. Install the required production dependencies:
pip install streamlit joblib lightgbm scikit-learn pandas numpy

4. Fire up the local application server:
python -m streamlit run app.py
