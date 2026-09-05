# Loan Approval Prediction

This project predicts whether a loan application is likely to be approved using a Logistic Regression classification model.

## Project Objective

The goal is to build a machine learning classification pipeline that can:

- preprocess applicant data
- handle missing values
- encode categorical features
- scale numeric features
- train a Logistic Regression model
- evaluate model performance
- analyze prediction probabilities
- save the trained model
- load the model for new predictions

## Dataset

The dataset contains applicant information such as:

- Gender
- Marital Status
- Dependents
- Education
- Self Employment
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

The target variable is:

`Loan_Status`

where:

- `Y` = Loan Approved
- `N` = Loan Not Approved

## Project Structure

```text
logistic_loan_prediction/
├── src/
│   ├── loan_analysis.py
│   └── predict_loan.py
├── data/
├── models/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore