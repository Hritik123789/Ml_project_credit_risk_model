# Ml_project_credit_risk_model
Credit Risk Model - Loan Default Prediction
🎯 Project Overview
This project implements an end-to-end Credit Risk Modeling system to predict loan defaults for financial institutions. Built using advanced machine learning techniques, the model analyzes customer financial data, loan details, and credit bureau information to assess default risk and calculate credit scores.​

Live Application: https://ml-project-credit-risk-model-1.streamlit.app/

📊 Dataset Information
The project uses three main datasets totaling 50,000 records:​

Customers Dataset (50,000 × 12): Demographics, income, employment status, residence details

Loans Dataset (50,000 × 15): Loan purpose, type, amounts, tenure, and financial metrics

Bureau Data (50,000 × 8): Credit history including open/closed accounts, delinquency information

Target Variable: Binary classification predicting loan default (0: Non-default, 1: Default) with class imbalance (91.4% vs 8.6%).​

🔧 Technical Architecture
Data Processing Pipeline
Data Cleaning:

Handled missing values in residence_type using mode imputation

Removed outliers where processing fee exceeded 3% of loan amount

Fixed data entry errors (e.g., 'Personaal' → 'Personal')

Applied business validation rules for GST and disbursement amounts​

Feature Engineering:
Three key derived features were created:​

Loan-to-Income Ratio (LTI): loan_amount / income

Delinquency Ratio: (delinquent_months × 100) / total_loan_months

Average DPD per Delinquency: total_dpd / delinquent_months

Feature Selection
VIF Analysis: Removed multicollinear features (sanction_amount, processing_fee, gst, net_disbursement) with VIF > 10.​

Information Value (IV) Analysis:
Selected 10 features with IV > 0.02:​​

credit_utilization_ratio (IV: 2.353)

delinquency_ratio (IV: 0.717)

loan_to_income (IV: 0.476)

avg_dpd_per_delinquency (IV: 0.402)

loan_purpose (IV: 0.369)

residence_type (IV: 0.247)

loan_tenure_months (IV: 0.219)

loan_type (IV: 0.163)

age (IV: 0.089)

number_of_open_accounts (IV: 0.085)

🤖 Model Development
Multiple Approaches Tested
Attempt 1 - Baseline Models (No Class Imbalance Handling):

Logistic Regression: F1-score 0.78

Random Forest: F1-score 0.78

XGBoost: F1-score 0.79​

Attempt 2 - Random Under Sampling:

Logistic Regression: Recall 0.96, Precision 0.51

XGBoost: Recall 0.99, Precision 0.52​

Attempt 3 - SMOTE-Tomek + Optuna Tuning (Logistic Regression):

Best F1-score: 0.946

Hyperparameters: C=9.37, solver='saga', tol=0.018​

Attempt 4 - SMOTE-Tomek + Optuna Tuning (XGBoost):

Best F1-score: 0.976

Key hyperparameters: max_depth=10, eta=0.264, lambda=1.88​

Final Model Selection
Logistic Regression with SMOTE-Tomek was chosen as the production model for superior interpretability.​

Performance Metrics:

Accuracy: 93%

Precision (Class 1): 0.57

Recall (Class 1): 0.94

F1-Score (Class 1): 0.71

AUC-ROC: 0.984

Gini Coefficient: 0.967

KS Statistic: 85.98 (at Decile 8)​

📈 Model Evaluation
Feature Importance
The model coefficients reveal the most influential predictors:​

loan_to_income: +18.10 (highest positive impact)

credit_utilization_ratio: +16.18

delinquency_ratio: +13.93

loan_purpose (Home): -3.70 (protective factor)

Risk Segmentation
Decile Analysis shows excellent rank ordering:​

Decile 9 (Highest Risk): 72% event rate

Decile 8: 12.7% event rate

Deciles 5-0: 0% event rate

The KS statistic of 85.98 indicates exceptional model discrimination capability.​

🚀 Deployment
Streamlit Web Application
The model is deployed as an interactive web application using Streamlit:​​

Features:

Real-time credit risk prediction

User-friendly input forms for customer and loan data

Instant default probability calculation

Credit score generation

Visual risk assessment dashboard

Model Artifacts:

python
# Saved components
- finalmodel (Logistic Regression)
- scaler (MinMaxScaler)
- feature_names
- cols_to_scale
💻 Installation & Usage
Prerequisites
bash
pip install pandas numpy scikit-learn xgboost
pip install streamlit optuna imbalanced-learn
pip install matplotlib seaborn statsmodels
Running Locally
bash
# Clone the repository
git clone [your-repo-url]

# Navigate to project directory
cd credit-risk-model

# Run Streamlit app
streamlit run app.py
Making Predictions
python
# Load the model
from joblib import load
model_data = load('artifacts/model_data.joblib')

# Prepare input data with required features
# Make prediction
probability = model_data['finalmodel'].predict_proba(input_data)
📁 Project Structure
text
credit-risk-model/
├── dataset/
│   ├── customers.csv
│   ├── loans.csv
│   └── bureau_data.csv
├── artifacts/
│   └── model_data.joblib
├── app.py                  # Streamlit application
├── model.py               # Model training script
├── prediction.py          # Prediction functions
├── requirements.txt
└── README.md
🔍 Key Insights
Credit Utilization and Loan-to-Income Ratio are the strongest default predictors​

Home loans show significantly lower default rates compared to personal/auto loans​

Delinquency history is a critical risk indicator​

Younger borrowers show slightly higher default propensity​

The model achieves 94% recall for defaults, minimizing false negatives​

📊 Business Value
Risk Assessment Automation: Instant credit decisions

Loss Reduction: Identify 94% of potential defaults

Portfolio Optimization: Data-driven lending strategies

Regulatory Compliance: Transparent, explainable model aligned with Basel II requirements​

Scalability: Handles high-volume applications efficiently

🛠️ Technologies Used
Python 3.10+

Machine Learning: scikit-learn, XGBoost, imbalanced-learn

Hyperparameter Optimization: Optuna

Data Processing: pandas, numpy

Visualization: matplotlib, seaborn

Deployment: Streamlit

Model Persistence: joblib

📝 Model Limitations
Trained on specific demographic and loan data distributions

Performance may degrade with data drift over time

Requires periodic retraining with recent data

Class imbalance handling may affect precision in production

🔄 Future Enhancements
Implement Population Stability Index (PSI) monitoring

Add SHAP explanations for individual predictions

Develop A/B testing framework for model versions

Integration with loan origination systems

Mobile-responsive dashboard


🙏 Acknowledgments
Dataset inspired by Codebasics ML Course​

Streamlit community for deployment resources​

Credit risk modeling best practices from industry standards​

