# Loan Approval Prediction

This project predicts whether a loan application is likely to be approved
using machine learning classification models.

The project compares Logistic Regression, Decision Tree, and Random Forest
models and selects the best-performing model based on evaluation metrics.

## Project Objective

The goal is to build a machine learning classification pipeline that can:

- preprocess applicant data
- handle missing values
- encode categorical features
- scale numeric features where required
- train multiple classification models
- evaluate model performance
- compare Logistic Regression, Decision Tree, and Random Forest
- analyze prediction probabilities
- select the best-performing model
- save the selected trained model
- load the saved model for predictions on new applicants

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
│
├── src/
│   ├── loan_analysis.py
│   └── predict_loan.py
│
├── data/
├── models/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Data Preprocessing

The project uses preprocessing pipelines to prepare the data before
model training.

### Numeric Features

Numeric features include:

- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term

Missing numeric values are handled using median imputation.

For Logistic Regression, numeric features are also standardized using
`StandardScaler`.

### Binary Features

`Credit_History` is treated as a binary feature.

Missing values are handled using the most frequent value.

### Categorical Features

Categorical features include:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- Property_Area

Missing categorical values are handled using the most frequent value.

Categorical features are converted into numeric features using
`OneHotEncoder`.

## Train-Test Split

The dataset is divided into training and test datasets using an 80/20 split.

Stratified sampling is used so that the proportion of approved and
non-approved loans remains similar in both datasets.

The test dataset is kept separate from model training and hyperparameter
selection and is used for final model evaluation.

## Models Evaluated

### Logistic Regression

Logistic Regression is used as the first classification model and baseline.

The preprocessing and Logistic Regression classifier are combined using
a scikit-learn `Pipeline`.

The model generates both:

- class predictions using `predict()`
- approval probabilities using `predict_proba()`

### Decision Tree

A Decision Tree classifier is trained using preprocessing suitable for
tree-based models.

Unlike Logistic Regression, feature scaling is not required for the
Decision Tree.

The `max_depth` hyperparameter is evaluated using `GridSearchCV` with
cross-validation.

The best Decision Tree configuration is then evaluated on the held-out
test dataset.

### Random Forest

A Random Forest classifier is trained as an ensemble of multiple Decision
Trees.

The model uses:

- 100 Decision Trees
- bootstrap sampling
- random feature selection during tree construction
- `random_state=42` for reproducibility

Random Forest uses the same tree-based preprocessing approach and does
not require feature scaling.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- ROC-AUC

These metrics provide different views of model performance and help avoid
selecting a model based only on accuracy.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8618 | 0.8400 | 0.9882 | 0.9081 | 0.8523 |
| Decision Tree | 0.8455 | 0.8235 | 0.9882 | 0.8984 | 0.7375 |
| Random Forest | 0.8211 | 0.8462 | 0.9059 | 0.8750 | 0.7796 |

## Final Model Selection

Logistic Regression was selected as the final model because it achieved
the strongest overall performance among the evaluated models.

It achieved:

- Accuracy: 86.18%
- Precision: 84.00%
- Recall: 98.82%
- F1 Score: 90.81%
- ROC-AUC: 85.23%

Although Random Forest achieved slightly higher precision, Logistic
Regression achieved better overall Accuracy, Recall, F1 Score, and
ROC-AUC in this experiment.

This demonstrates that a more complex machine learning algorithm does
not automatically produce better results. Model selection should be
based on evaluation results and the requirements of the business problem.

## Prediction on New Applicants

The selected Logistic Regression model is saved after training.

The `src/predict_loan.py` script loads the saved model and can generate
predictions for new applicant data.

The prediction pipeline automatically applies the same preprocessing
steps used during model training before generating the final prediction.

Example output:

```text
Prediction: Y
Probability of Loan Approval: 0.8766
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Git
- GitHub

## Key Machine Learning Concepts Demonstrated

This project demonstrates:

- supervised machine learning
- binary classification
- train-test splitting
- stratified sampling
- missing-value imputation
- categorical feature encoding
- feature scaling
- scikit-learn pipelines
- Logistic Regression
- Decision Trees
- Random Forest
- hyperparameter tuning
- cross-validation
- GridSearchCV
- probability prediction
- classification evaluation metrics
- ROC-AUC
- model comparison
- model selection
- model persistence
- prediction on unseen applicant data

## Conclusion

Three classification algorithms were evaluated for the loan approval
prediction problem: Logistic Regression, Decision Tree, and Random Forest.

Based on the current experiment, Logistic Regression produced the
strongest overall test performance and was selected as the final model.

The project demonstrates an end-to-end machine learning workflow from
data preprocessing and model training through evaluation, comparison,
model selection, and prediction on new data.