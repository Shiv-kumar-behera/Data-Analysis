from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np

# Loading Data
df = pd.read_csv("credit_score.csv")
df["Credit_Score"] = df["Credit_Score"].replace("Standard", "Good")

# Preprocessing
categories = ["Poor", "Good"]
encoder = OrdinalEncoder(categories=[categories])
df["Credit_Score"] = encoder.fit_transform(df[["Credit_Score"]])
df["Credit_Score"] = df["Credit_Score"].astype("int")

categories = ["Bad", "Standard", "Good"]
encoder = OrdinalEncoder(categories=[categories])
df["Credit_Mix"] = encoder.fit_transform(df[["Credit_Mix"]])
df["Credit_Mix"] = df["Credit_Mix"].astype("int")

categories = ["Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments"
]
encoder = OrdinalEncoder(categories=[categories])
df["Payment_Behaviour"] = encoder.fit_transform(df[["Payment_Behaviour"]])
df["Payment_Behaviour"] = df["Payment_Behaviour"].astype("int")

df["Occupation"] = LabelEncoder().fit_transform(df["Occupation"])

df["Payment_of_Min_Amount"] = LabelEncoder().fit_transform(df["Payment_of_Min_Amount"])

# Train and Test Datasets
X = df.drop(["Credit_Score"], axis=1)
y = df["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Model Fitting
model = RandomForestClassifier(n_estimators=10, random_state=40, oob_score=True)
model.fit(X_train, y_train)

# KS satistics
y_pred = model.predict_proba(X_test)
ks_df = pd.DataFrame({
    "p": y_pred[:,1],
    "y": y_test
})
ks, p_value = ks_2samp(ks_df.loc[ks_df.y==0,"p"], ks_df.loc[ks_df.y==1,"p"])
print(ks, p_value)