import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve

df = pd.read_csv("Finance.csv")

print(df.info())

df["Gender"].mode()[0]

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

df["Married"].mode()[0]

df["Married"] = df["Married"].fillna(df["Married"].mode()[0])

df["Dependents"].mode()[0]

df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])

df["Self_Employed"].mode()[0]

df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

df["LoanAmount"].median()

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

df["Loan_Amount_Term"].median()

df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

df["Credit_History"].median()

df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].median())

print(df.isna().sum()*100/len(df))

df.replace({
    "Loan_Status": {'N': 0, 'Y': 1},
    "Gender": {'Male': 0, 'Female': 1},
    "Education": {'Not Graduate': 0, 'Graduate': 1},
    "Married": {'No': 0, 'Yes': 1},
    "Self_Employed": {'No': 0, 'Yes': 1}
}, inplace=True)

def train_test_split_and_features(df):
    y = df["Loan_Status"]
    x = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    x = pd.get_dummies(data = x, columns = ["Property_Area","Dependents"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 0)
    print(x.head(5))
    print(x.columns)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test,features

x_train, x_test, y_train, y_test,features = train_test_split_and_features(df)

def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    random_forest = RandomForestClassifier(
                                            random_state=42,\
                                            max_depth=3,\
                                            min_samples_split= 0.01,\
                                            max_features= 0.8,
                                            max_samples= 0.8,
                                            class_weight="balanced")


    model = random_forest.fit(x_train, y_train)
    random_forest_predict = random_forest.predict(x_test)
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    print("confussion matrix")
    print(random_forest_conf_matrix)
    print("\n")
    print("Accuracy of Random Forest:",random_forest_acc_score*100,'\n')
    print(classification_report(y_test,random_forest_predict))
    return model

model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)

importances = pd.DataFrame(model.feature_importances_)
importances['features'] = features
importances.columns = ['importance','feature']
importances.sort_values(by = 'importance', ascending= True,inplace=True)

import matplotlib.pyplot as plt
plt.barh(importances.feature, importances.importance)

print(x_test.columns)

# -------------------------
# 2. Hardcoded test input (numeric)
# -------------------------


# EXACT column order from your x_train
columns = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area_Rural',
    'Property_Area_Semiurban',
    'Property_Area_Urban',
    'Dependents_0',
    'Dependents_1',
    'Dependents_2',
    'Dependents_3+'
]

# HARD CODED NUMERIC INPUT (One full row)
values = [
    0,      # Gender: Male
    0,      # Married: No
    1,      # Education: Graduate
    0,      # Self_Employed: No
    4000,   # ApplicantIncome
    0,      # CoapplicantIncome
    90,     # LoanAmount
    360,    # Loan_Amount_Term
    1,      # Credit_History

    # Property_Area: Semiurban
    0,      # Rural
    1,      # Semiurban
    0,      # Urban

    # Dependents = 1
    0,      # 0
    1,      # 1
    0,      # 2
    0       # 3+
]


# Build DataFrame
df = pd.DataFrame([values], columns=columns)

# Predict
pred = model.predict(df)[0]

# Output
if pred == 1:
    print("Loan Status: APPROVED ✔")
else:
    print("Loan Status: NOT APPROVED ❌")

# At the end of your model training code
import pickle

# assuming your model object is 'model'
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")


pickle.dump(list(x_train.columns), open("model_features.pkl", "wb"))
print("Feature names saved as model_features.pkl")