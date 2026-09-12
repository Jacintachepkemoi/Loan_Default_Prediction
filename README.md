# Loan_Default_Prediction
End-to-End Loan Default Risk Prediction Pipeline

A comprehensive machine learning architecture designed to evaluate credit risk and predict loan defaults using a large-scale financial dataset containing 255,347 unique borrower records.https://www.kaggle.com/datasets/nikhil1e9/loan-default?resource=download


The Business Challenge and Strategy

In consumer lending, a False Negative—approving a loan for an applicant who will ultimately default—is devastatingly expensive for a bank. This portfolio project explicitly shifts focus away from traditional overall accuracy metrics to navigate the critical Precision-Recall trade-off. The primary objective is to optimize the model's default-catching capabilities (Recall) to serve as a defensive financial shield during high-risk economic climates.

Data Engineering and Preprocessing Steps

High-Cardinality Identifier Exclusion:The unique customer string column (LoanID) was dropped from the dataset prior to transformation. This step was mandatory to prevent a feature explosion bug during encoding that would have generated over 255,000 empty binary columns and crashed the virtual memory environment.

Targeted Category One-Hot Encoding:Pandas get_dummies was configured with dtype=int and drop_first=True, applied strictly to the true text-based categorical columns (Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner). This successfully transformed text labels into readable binary flags while leaving numeric variables entirely intact.

Input Feature Scaling:Continuous numerical fields (Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio) were normalized using StandardScaler. This centered the distribution of each column around a mean of 0 with a standard variance of 1, preventing high-magnitude columns like Income from mathematically overpowering smaller counters like CreditScore or Age.

Class Imbalance Diagnostics:Exploratory data analysis revealed a severe class imbalance in the target variable (Default). Paid-back loans made up 88.39% of the dataset (225,694 rows), while actual defaults comprised only 11.61% (29,653 rows).

Imbalance Resolution via Dual-Balancing:To prevent the models from falling into a lazy bias of predicting non-default for every applicant, the training dataset split was processed using SMOTE (Synthetic Minority Over-sampling Technique). This generated synthetic variations of minority rows to build a perfectly balanced 50/50 baseline in memory. Furthermore, the tree models were augmented with a custom class-weight multiplier (scale_pos_weight=7.6), derived directly from the raw imbalance ratio, to aggressively punish the algorithms for missing rare default patterns.


The 4-Model Tournament Leaderboard

Four distinct competitive frameworks were trained on an isolated 80/20 train-test split. To guarantee unbiased evaluation, all probability calculations (y_proba) and absolute classification choices (y_pred) were generated strictly on the hidden test set (X_test), simulating how the models would perform on fresh real-world applicants.

LightGBM Classifier (SMOTE + 7.6 Weight)ROC AUC Score: 0.7269Class 1 Recall: 76.0%Class 1 Precision: 19.0%Status: Deployed Portfolio Champion

XGBoost Classifier (SMOTE + 7.6 Weight)ROC AUC Score: 0.7200Class 1 Recall: 75.0%Class 1 Precision: 19.0%Status: Runner-Up Contender

Random Forest Classifier (SMOTE Baseline)ROC AUC Score: 0.7043Class 1 Recall: 12.0%Class 1 Precision: 27.0%Status: Eliminated (Too Conservative)

Logistic Regression (SMOTE Baseline)ROC AUC Score: 0.6689Class 1 Recall: 25.0%Class 1 Precision: 23.0%Status: Eliminated (Weakest Separability)


Core Architectural and Strategic Insights

The Impact of Weight Multipliers:
Standard bagging models (Random Forest) and basic gradient boosters without custom tuning stalled at an unacceptable 12% to 14% Recall rate due to the heavy dataset imbalance. Introducing the 7.6 class penalty modifier radically altered the mathematical decision boundaries. By forcing the algorithms to treat a default row as 7.6 times more important than a normal row, the default-catching capability (Recall) skyrocketed to a massive 76.0%, successfully intercepting 4,455 out of 5,900 defaults.

Justification of Deployed Model:
LightGBM with dual SMOTE and class-weight tuning was selected as the definitive portfolio champion. It achieved the highest overall sorting power (0.7269 ROC AUC) and maximum risk protection. While this aggressive optimization causes a drop in precision (19%) due to false alarms, the model acts as an ideal defensive asset for financial institutions where preventing toxic debt overrides the cost of manual applicant review.

Explainable AI (XAI) Compliance:
To remove the black-box limitation of gradient boosting, the pipeline implements SHAP (SHapley Additive exPlanations) values grounded in cooperative game theory. By evaluating localized features for individual applicants, the system calculates exact positive and negative contribution metrics for every decision. A review of sample waterfall charts demonstrates that features such as a PhD status, home loan intent, and low loan terms serve as dominant risk reducers, successfully overriding risk increasers like high interest rates to prove mathematical compliance for fair-lending regulatory audits.


Feel free to browse the repository file tree or launch the Loan_Prediction_Pipeline notebook directly in Google Colab using the interactive badge embedded at the top of the file.
