import pandas as pd
import joblib
loaded_model=joblib.load("models/loan_logistic_model.pkl")
new_applicant=pd.DataFrame([{"Gender":"Male",
                             "Married":"Yes",
                             "Dependents":"0",
                             "Education":"Graduate",
                             "Self_Employed":"No",
                             "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500.0,
        "LoanAmount": 120.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"}])
prediction=loaded_model.predict(new_applicant)
probabilities=loaded_model.predict_proba(new_applicant)
print("\n Predictions")
print(prediction[0])
print("\n Predict probability")
print(probabilities[0][1])
classes=loaded_model.named_steps["classifier"].classes_
print("\n Model classes")
print(classes)
for classname,probability in zip(classes,probabilities[0]):
    print(f"Probability of class name {classname} is {probability:.4f}")
