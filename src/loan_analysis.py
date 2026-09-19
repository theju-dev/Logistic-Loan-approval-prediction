import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
print("\n Training target distribution")
print(y_train.value_counts())
print("\n Training target percentage")
print((y_train.value_counts(normalize=True)*100).round(2))
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
model=Pipeline(steps=[("preprocessor",preprocesssor),("classifier",LogisticRegression(max_iter=100,random_state=42))])
balanced_logistic_model=Pipeline(steps=[("preprocessor",preprocesssor),("classifier",LogisticRegression(max_iter=1000,random_state=42,class_weight="balanced"))])
balanced_logistic_model.fit(X_train,y_train)
balanced_logistic_predictions=balanced_logistic_model.predict(X_test)
balanced_logistic_probabilties=balanced_logistic_model.predict_proba(X_test)
balanced_logistic_classes=balanced_logistic_model.named_steps["classifier"].classes_
balanced_logistic_positive_class_index=list(balanced_logistic_classes).index("Y")
balanced_logistic_positive_probabilties=balanced_logistic_probabilties[:,balanced_logistic_positive_class_index]
balanced_logistic_accuracy=accuracy_score(y_test,balanced_logistic_predictions)
balanced_logistic_precision=precision_score(y_test,balanced_logistic_predictions,pos_label="Y")
balanced_logistic_recall=recall_score(y_test,balanced_logistic_predictions,pos_label="Y")
balanced_logistic_f1=f1_score(y_test,balanced_logistic_predictions,pos_label="Y")
y_test_binary=(y_test=="Y").astype(int)
balanced_logistic_roc_auc=roc_auc_score(y_test_binary,balanced_logistic_positive_probabilties)
balanced_logistic_confusion_matrix=confusion_matrix(y_test,balanced_logistic_predictions,labels=["N","Y"])
print("\nBalanced Logistic Regression Evalaution")
print(f"Accuracy:{balanced_logistic_accuracy*100:.2f}%")
print(f"Precision:{balanced_logistic_precision*100:.2f}%")
print(f"Recall:{balanced_logistic_recall*100:.2f}%")
print(
    f"F1 Score: {balanced_logistic_f1 * 100:.2f}%"
)
print(
    f"ROC-AUC: {balanced_logistic_roc_auc * 100:.2f}%"
)

print("\nBalanced Logistic Regression Confusion Matrix")
print(balanced_logistic_confusion_matrix)
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

print("\n START WITH DECISION TREE CLASSIFIER ALGORITHM")
tree_numeric_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="median"))])
tree_binary_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent"))])
tree_categorical_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                             ("encoder",OneHotEncoder(handle_unknown="ignore"))])
tree_preprocessor=ColumnTransformer(transformers=[("num",tree_numeric_transformer,numeric_features),
                                                  ("binary",tree_binary_transformer,binary_features),
                                                  ("cat",tree_categorical_transformer,categorical_features)])
tree_model=Pipeline(steps=[("preprocessor",tree_preprocessor),("classifier",DecisionTreeClassifier(max_depth=4,random_state=42))])
tree_model.fit(X_train,y_train)
tree_pred=tree_model.predict(X_test)
tree_pred_proba=tree_model.predict_proba(X_test)
#tree_prob_y=tree_pred_proba[:,1]
print(tree_pred)
print(tree_pred_proba)
print("\n decision tree classes")
tree_classes=tree_model.named_steps["classifier"].classes_
tree_y_index=list(tree_classes).index("Y")
tree_prob_y=tree_pred_proba[:,tree_y_index]
print("\n Tree prob y")
print(tree_prob_y)
tree_accuracy=accuracy_score(y_test,tree_pred)
tree_precision=precision_score(y_test,tree_pred,pos_label="Y")
tree_recall=recall_score(y_test,tree_pred,pos_label="Y")
tree_f1=f1_score(y_test,tree_pred,pos_label="Y")
tree_confusion_matrix=confusion_matrix(y_test,tree_pred)
tree_roc_auc=roc_auc_score(y_test_binary,tree_prob_y)
print("\n ====Decision tree results")
print("\nAccuracy:",tree_accuracy)
print("\nPrecision:",tree_precision)
print("\nrecall:",tree_recall)
print("\nf1score:",tree_f1)
print("\nconfusionmatrix:",tree_confusion_matrix)
print("\nrocauc:",tree_roc_auc)
print("Classification report:",classification_report(y_test,tree_pred))
tree_parm_grid={"classifier__max_depth":[2,3,4,5,6]}
tree_grid_search=GridSearchCV(estimator=tree_model,param_grid=tree_parm_grid,cv=5,scoring="roc_auc")
tree_grid_search.fit(X_train,y_train)
print("\n===Decision tree grid search")
print("Best parameter")
print(tree_grid_search.best_params_)
print("\n Best cross validation roc-auc")
print(tree_grid_search.best_score_)
best_tree_model=tree_grid_search.best_estimator_
print("\nBest model")
print(best_tree_model)
best_tree_pred=best_tree_model.predict(X_test)
print(best_tree_pred)
best_tree_proba=best_tree_model.predict_proba(X_test)
print(best_tree_proba)
best_tree_proba_y=best_tree_proba[:,1]
best_tree_accuracy=accuracy_score(y_test,best_tree_pred)
best_tree_precision=precision_score(y_test,best_tree_pred,pos_label="Y")
best_tree_recall=recall_score(y_test,best_tree_pred,pos_label="Y")
best_tree_f1=f1_score(y_test,best_tree_pred,pos_label="Y")
best_tree_confusion_matrix = confusion_matrix(
    y_test,
    best_tree_pred
)

best_tree_roc_auc = roc_auc_score(
    y_test_binary,
    best_tree_proba_y
)
print("\n===== TUNED DECISION TREE TEST RESULTS =====")

print("Accuracy:", best_tree_accuracy)
print("Precision:", best_tree_precision)
print("Recall:", best_tree_recall)
print("F1 Score:", best_tree_f1)
print("ROC-AUC:", best_tree_roc_auc)

print("\nConfusion Matrix:")
print(best_tree_confusion_matrix)
print("\n classification report")
print(classification_report(y_test,best_tree_pred))
print("\n Random Forest")
random_forest_model=Pipeline(steps=[("preprocessor",tree_preprocessor),("classifier",RandomForestClassifier(n_estimators=100,random_state=42))])
random_forest_model.fit(X_train,y_train)
rf_classifier=random_forest_model.named_steps["classifier"]
print("Number of trees:")
print(len(rf_classifier.estimators_))
random_forest_pred=random_forest_model.predict(X_test)
random_forest_proba=random_forest_model.predict_proba(X_test)
random_forest_classes=random_forest_model.named_steps["classifier"].classes_
random_forest_y_index=list(random_forest_classes).index("Y")
random_forest_y_proba=random_forest_proba[:,random_forest_y_index]
'''print("\nRandom Forest classes:")
print(random_forest_classes)

print("\nFirst 5 predictions:")
print(random_forest_pred[:5])

print("\nFirst 5 probabilities:")
print(random_forest_proba[:5])'''
random_forest_accuracy=accuracy_score(y_test,random_forest_pred)
random_forest_precision=precision_score(y_test,random_forest_pred,pos_label="Y")
random_forest_recall=recall_score(y_test,random_forest_pred,pos_label="Y")
random_forest_f1=f1_score(y_test,random_forest_pred,pos_label="Y")
random_forest_roc_auc=roc_auc_score(y_test_binary,random_forest_y_proba)
random_forest_confusion_matrix=confusion_matrix(y_test,random_forest_pred)
print("\n===== RANDOM FOREST RESULTS =====")

print("Accuracy:", random_forest_accuracy)
print("Precision:", random_forest_precision)
print("Recall:", random_forest_recall)
print("F1 Score:", random_forest_f1)
print("ROC-AUC:", random_forest_roc_auc)
print("\nConfusion Matrix:")
print(random_forest_confusion_matrix)
print("\nclassification report")
print(classification_report(y_test,random_forest_pred))
X_train_transformed=random_forest_model.named_steps["preprocessor"].transform(X_train)
print("\n===Random forest internal check=====")
print("Number of trees")
print(random_forest_model.named_steps["classifier"].estimators_)
print(len(random_forest_model.named_steps["classifier"].estimators_))
print("Bootstrap enabled")
print(random_forest_model.named_steps["classifier"].bootstrap)
print("\n original training shape")
print(X_train.shape)
print("Transformed training shape")
print(X_train_transformed.shape)
print("\n Max features setting")
print(random_forest_model.named_steps["classifier"].max_features)
first_tree=random_forest_model.named_steps["classifier"].estimators_[0]
print("Number of features considered at each split")
print(first_tree.max_features_)
'''model_comparision=pd.DataFrame({"Model":["Logistic Regression","Decision Tree","Random Forest"],
                                "Accuracy":[accuracy,best_tree_accuracy,random_forest_accuracy],
                                "Precision":[precision,best_tree_precision,random_forest_precision],
                                "Recall":[recall,best_tree_recall,random_forest_recall],
                                "f1score":[f1score,best_tree_f1,random_forest_f1],
                                "ROC-AUC":[roc_auc,best_tree_roc_auc,random_forest_roc_auc]})
print("\n===model comparision=====")
print(model_comparision)
print("\n===Final model selection=====")
print("selected model:Logistic Regression")
print("Reason: Logistic Regression achieved best overall performace across accuracy , f1 score, roc-auc")
'''

print("\n====KNearest Neighbors=====")
knn_model=Pipeline(steps=[("preprocessor",preprocesssor),("classifier",KNeighborsClassifier(n_neighbors=5))])
'''knn_model.fit(X_train,y_train)
knn_pred=knn_model.predict(X_test)
knn_proba=knn_model.predict_proba(X_test)
knn_classes=knn_model.named_steps["classifier"].classes_
knn_y_index=list(knn_classes).index("Y")
knn_prob_y=knn_proba[:,knn_y_index]
knn_accuracy=accuracy_score(y_test,knn_pred)
knn_precision=precision_score(y_test,knn_pred,pos_label="Y")
knn_recall=recall_score(y_test,knn_pred,pos_label="Y")
knn_f1=f1_score(y_test,knn_pred,pos_label="Y")
knn_roc_auc=roc_auc_score(y_test_binary,knn_prob_y)
knn_confusion_matrix=confusion_matrix(y_test,knn_pred)
print("Accuracy:",knn_accuracy)
print("Precision:",knn_precision)
print("recall:",knn_recall)
print("F1 Score:", knn_f1)
print("ROC-AUC:", knn_roc_auc)
print("confusion matrix:",knn_confusion_matrix)
print("\n classification report:")
print(classification_report(y_test,knn_pred))'''
knn_param_grid={"classifier__n_neighbors":[3,5,7,9,11]}
knn_grid_search=GridSearchCV(estimator=knn_model,param_grid=knn_param_grid,cv=5,scoring="roc_auc")
knn_grid_search.fit(X_train,y_train)
print("Best parameters:",knn_grid_search.best_params_)
print("Best CV ROC-AUC:",knn_grid_search.best_score_)
best_knn_model=knn_grid_search.best_estimator_
knn_pred=best_knn_model.predict(X_test)
knn_proba=best_knn_model.predict_proba(X_test)
knn_classes=best_knn_model.named_steps["classifier"].classes_
knn_y_index=list(knn_classes).index("Y")
knn_prob_y=knn_proba[:,knn_y_index]
knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

knn_precision = precision_score(
    y_test,
    knn_pred,
    pos_label="Y"
)

knn_recall = recall_score(
    y_test,
    knn_pred,
    pos_label="Y"
)

knn_f1 = f1_score(
    y_test,
    knn_pred,
    pos_label="Y"
)

knn_roc_auc=roc_auc_score(y_test_binary,knn_prob_y)
knn_confusion_matrix = confusion_matrix(
    y_test,
    knn_pred
)

# 11. Print evaluation
print("\nKNN Test Results")

print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)
print("ROC-AUC:", knn_roc_auc)

print("\nConfusion Matrix:")
print(knn_confusion_matrix)
print("\nClassification Report:")
print(classification_report(y_test,knn_pred))
'''/*model_comparision=pd.DataFrame({"Model":["Logistic Regression","Decision Tree","Random Forest","Tuned KNN"],
                                "Accuracy":[accuracy,best_tree_accuracy,random_forest_accuracy,knn_accuracy],
                                "Precision":[precision,best_tree_precision,random_forest_precision,knn_precision],
                                "Recall":[recall,best_tree_recall,random_forest_recall,knn_recall],
                                "f1score":[f1score,best_tree_f1,random_forest_f1,knn_f1],
                                "ROC-AUC":[roc_auc,best_tree_roc_auc,random_forest_roc_auc,knn_roc_auc]})
print(model_comparision.round(4))
comparision_display=model_comparision.copy()
metric_columns=["Accuracy","Precision","Recall","f1score","ROC-AUC"]
comparision_display[metric_columns]=(comparision_display[metric_columns]*100).round(2)
print(comparision_display)
comparision_display=comparision_display.rename(columns={column:f"{column} (%)" for column in metric_columns})
print(comparision_display)
print("\n Model comparision")
print(comparision_display.to_string(index=False))'''
svm_model=Pipeline(steps=[("preprocessor",preprocesssor),("classifier",SVC())])
svm_param_grid=[{"classifier__kernel":["linear"],
                 "classifier__C":[0.1,1,10]},
                 {"classifier__kernel":["rbf"],
                  "classifier__C":[0.1,1,10],
                  "classifier__gamma":["scale",0.01,0.1,1]}]
svm_grid_search=GridSearchCV(estimator=svm_model,param_grid=svm_param_grid,cv=5,scoring="roc_auc",n_jobs=-1)
svm_grid_search.fit(X_train,y_train)
print("\n Best SVM parameters")
print(svm_grid_search.best_params_)
print("\n Best SVM CV ROC-AUC")
print(svm_grid_search.best_score_)
print("\n Best svm ESTIMATOR")
best_svm_model=svm_grid_search.best_estimator_
svm_predictions=best_svm_model.predict(X_test)
svm_decision_scores=best_svm_model.decision_function(X_test)
svm_classes=best_svm_model.named_steps["classifier"].classes_
'''print("\n SVM classes")
print(svm_classes)'''
if svm_classes[1]=="Y":
    svm_positive_scores=svm_decision_scores
elif svm_classes[0]=="Y":
    svm_positive_scores=-svm_decision_scores
else:
    raise ValueError("Positive Y class was not found in SVM classes")
svm_accuracy=accuracy_score(y_test,svm_predictions)
svm_precision=precision_score(y_test,svm_predictions,pos_label="Y")
svm_recall=recall_score(y_test,svm_predictions,pos_label="Y")
svm_f1=f1_score(y_test,svm_predictions,pos_label="Y")
svm_roc_auc=roc_auc_score(y_test_binary,svm_positive_scores)
print("\n SVM Test Metrics")
print(f"Accuracy:{svm_accuracy*100:.2f}")
print(f"Precision:{svm_precision*100:.2f}")
print(f"Recall:{svm_recall*100:.2f}")
print(f"F1 score:{svm_f1*100:.2f}")
print(f"ROC-AUC:{svm_roc_auc*100:.2f}")
svm_confusion_matrix=confusion_matrix(y_test,svm_predictions,labels=["N","Y"])
print(svm_confusion_matrix)
print("\n SVM Classification report")
print(classification_report(y_test,svm_predictions,labels=["N","Y"],zero_division=0))
naive_bayes_categorical_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                                    ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
naive_bayes_preprocessor=ColumnTransformer(transformers=[("num",numeric_transformer,numeric_features),
                                                         ("binary",binary_transformer,binary_features),
                                                         ("cat",naive_bayes_categorical_transformer,categorical_features)])
naive_bayes_model=Pipeline(steps=[("preprocessor",naive_bayes_preprocessor),("classifier",GaussianNB())])

'''model_comparision=pd.DataFrame({"Model":["Logistic Regression","Decision Tree","Random Forest","Tuned KNN","Tuned SVM"],
                                "Accuracy":[accuracy,best_tree_accuracy,random_forest_accuracy,knn_accuracy,svm_accuracy],
                                "Precision":[precision,best_tree_precision,random_forest_precision,knn_precision,svm_precision],
                                "Recall":[recall,best_tree_recall,random_forest_recall,knn_recall,svm_recall],
                                "f1score":[f1score,best_tree_f1,random_forest_f1,knn_f1,svm_f1],
                                "ROC-AUC":[roc_auc,best_tree_roc_auc,random_forest_roc_auc,knn_roc_auc,svm_roc_auc]})

comparision_display=model_comparision.copy()
metric_columns=["Accuracy","Precision","Recall","f1score","ROC-AUC"]
comparision_display[metric_columns]=(comparision_display[metric_columns]*100).round(2)
comparision_display=comparision_display.rename(columns={column:f"{column} (%)" for column in metric_columns})
print("\n Final model comparision")
print(comparision_display.to_string(index=False))'''
naive_bayes_model.fit(X_train,y_train)
naive_bayes_predictions=naive_bayes_model.predict(X_test)
naive_bayes_probabilities=naive_bayes_model.predict_proba(X_test)
naive_bayes_classes=naive_bayes_model.named_steps["classifier"].classes_
naive_bayes_positive_class_index=list(naive_bayes_classes).index("Y")
naive_bayes_positive_probabilties=naive_bayes_probabilities[:,naive_bayes_positive_class_index]
naive_bayes_accuracy=accuracy_score(y_test,naive_bayes_predictions)
naive_bayes_precision=precision_score(y_test,naive_bayes_predictions,pos_label="Y",zero_division=0)
naive_bayes_recall=recall_score(y_test,naive_bayes_predictions,pos_label="Y",zero_division=0)
naive_bayes_f1=f1_score(y_test,naive_bayes_predictions,pos_label="Y",zero_division=0)
naive_bayes_roc_auc=roc_auc_score(y_test_binary,naive_bayes_positive_probabilties)
print("\nGaussian Naive Bayes Evalauation")
print(f"Accuracy:{naive_bayes_accuracy:.4f}")
print(f"Recall:{naive_bayes_recall:.4f}")
print(f"Precision:{naive_bayes_precision:.4f}")
print(f"F1 Score:{naive_bayes_f1:.4f}")
print(f"ROC-AUC:{naive_bayes_roc_auc:.4f}")
naive_bayes_confusion_matrix=confusion_matrix(y_test,naive_bayes_predictions,labels=["N","Y"])
print(f"Confusion matrix:{naive_bayes_confusion_matrix}")
print("\ngaussian naive bayes classification report")
print(classification_report(y_test,naive_bayes_predictions,labels=["N","Y"],zero_division=0))
model_comparision=pd.DataFrame({"Model":["Logistic Regression","Balanced Logistic Regression","Decision Tree","Random Forest","Tuned KNN","Tuned SVM","Gaussian Naive Bayes"],
                                "Accuracy":[accuracy,balanced_logistic_accuracy,best_tree_accuracy,random_forest_accuracy,knn_accuracy,svm_accuracy,naive_bayes_accuracy],
                                "Precision":[precision,balanced_logistic_precision,best_tree_precision,random_forest_precision,knn_precision,svm_precision,naive_bayes_precision],
                                "Recall":[recall,balanced_logistic_recall,best_tree_recall,random_forest_recall,knn_recall,svm_recall,naive_bayes_recall],
                                "f1score":[f1score,balanced_logistic_f1,best_tree_f1,random_forest_f1,knn_f1,svm_f1,naive_bayes_f1],
                                "ROC-AUC":[roc_auc,balanced_logistic_roc_auc,best_tree_roc_auc,random_forest_roc_auc,knn_roc_auc,svm_roc_auc,naive_bayes_roc_auc]})
comparision_display=model_comparision.copy()
metric_columns=["Accuracy","Precision","Recall","f1score","ROC-AUC"]
comparision_display[metric_columns]=(comparision_display[metric_columns]*100).round(2)
comparision_display=comparision_display.rename(columns={column:f"{column} (%)" for column in metric_columns})
print("\n Final model comparision")
print(comparision_display.to_string(index=False))