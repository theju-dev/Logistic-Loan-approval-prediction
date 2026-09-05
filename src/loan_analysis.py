import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,classification_report,precision_score,recall_score,f1_score
df=pd.read_csv("data/Loan_Data.csv")
print(df.head())
print("\n shape")
print(df.shape)
print("\n columns:")
print(df.columns.tolist())
print("\n data types:")
print(df.dtypes)
print("\n Info:")
df.info()
print("\n Mising values")
print(df.isnull().sum())
print("\n Duplicate rows")
print(df.duplicated().sum())
categorical_columns=["Gender","Married","Dependents","Education","Self_Employed","Property_Area","Loan_Status"]
for column in categorical_columns:
    print(f"Value counts - {column}")
    print(df[column].value_counts(dropna=False))
'''feature_columns=["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount",
                 "Loan_Amount_Term","Credit_History","Property_Area"]
target_column="Loan_Status"
X=df[feature_columns]   
y=df[target_column]
numeric_columns=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
print("\n Numeric columns")
print(df[numeric_columns].describe())'''
feature_columns=["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount",
                 "Loan_Amount_Term","Credit_History","Property_Area"]
target_column="Loan_Status"
X=df[feature_columns]
y=df[target_column]
numeric_features=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
binary_features=["Credit_History"]
categorical_features=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("\nTrain shape:")
print(X_train.shape)
print("\nTest shape:")
print(X_test.shape)
print("\nTrain target distribution")
print(y_train.value_counts(normalize=True))
print("\n Test target distribution")
print(y_test.value_counts(normalize=True))
numeric_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
binary_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent"))])
categorical_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore"))])
preprocesssor=ColumnTransformer(transformers=[("num",numeric_transformer,numeric_features),("binary",binary_transformer,binary_features),("cat",categorical_transformer,categorical_features)])
model=Pipeline(steps=[("preprocessor",preprocesssor),("classifier",LogisticRegression(max_iter=100))])
print("\n train columns:")
print(X_train.columns.tolist())
print("\n numeric features:")
print(numeric_features)
print("\n binary features:")
print(binary_features)
print("\n categoric features:")
print(categorical_features)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)
print(y_pred[:3])
print(y_prob[:3])
print("\n actual results")
print(y_test[:3])
print(model.named_steps["classifier"].classes_)
print("\n Loan status column first 5 rows")
print(y_train.head(5))
accuracy=accuracy_score(y_test,y_pred)  # how many predictions are correct out of all predictions , suppose there are 100 applicants, 
#out of 100 , assume model got 82 correct , 18 wrong so accuracy is 82/100 
precision=precision_score(y_test,y_pred,pos_label="Y") # out of all cases which model predicted as Y , how much were actually Y
# suppose model predicts Y for 20 applicants ,actual Y were 16 and remaining 4 were actually N , so precision 16/20=0.80
# precision = TP/(TP+FP)
recall=recall_score(y_test,y_pred,pos_label="Y") # out of all actual Y cases , how many did model predicts as Y
# recall = TP/(TP+FN)
f1score=f1_score(y_test,y_pred,pos_label="Y") #F1SCORE=(2*PRECISION*RECALL)/(PRECISION+RECALL)
conf_matrix=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
print("\n Model evaluation")
print("\n Accuracy")
print(accuracy)
print("\n Precision")
print(precision)
print("\n recall")
print(recall)
print("f1score")
print(f1score)
print("\n confusion matrix")
print(conf_matrix)
print("\n classifcation report")
print(report)
Y_prob_y=y_prob[:,1]
threshold=0.7
y_pred_70=["Y" if probability>=threshold else "N" for probability in Y_prob_y]
print("\n 0.7 threshold evaluation")
print("\n confusion matrix")
print("confusion matrix:",confusion_matrix(y_test,y_pred_70)
      )
print("Classification report")
print(classification_report(y_test,y_pred_70))
y_test_binary=(y_test=="Y").astype(int)
roc_auc=roc_auc_score(y_test_binary,Y_prob_y)
print("\nroc-auc score")
print(roc_auc)
results=X_test.copy()
results["Actual"]=y_test
results["Predicted"]=y_pred
results["Predicted_Y"]=Y_prob_y
print("\n results")
print(results)
errors=results[results["Actual"]!=results["Predicted"]]
print("\n Misclasssified applicants")
print(errors)
print(len(errors))
feature_names=model.named_steps["preprocessor"].get_feature_names_out()
coefficients=model.named_steps["classifier"].coef_[0]
print("\n Number of preprocessed featured")
print(len(feature_names))
print("\n Number of coefficients")
print(len(coefficients))
print("\n Feature names")
print(feature_names)
print("\n coefficients")
##print(coefficients[0])
coefficient_df=pd.DataFrame({"Feature":feature_names,
                            "Coefficient":coefficients})
print(coefficient_df)
coefficient_df=coefficient_df.sort_values(by="Coefficient",ascending=False)
print("\n coefficient_df sorted coefficients")
print(coefficient_df)
joblib.dump(model,"models/loan_logistic_model.pkl")
print("\n model saved successfully")
