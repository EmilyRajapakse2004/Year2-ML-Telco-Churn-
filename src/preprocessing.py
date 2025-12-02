# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_dataset(data):
    """Preprocess full dataset"""

    # Fix numeric column
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    # Encode target
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical
    encoders = {}
    cat_cols = data.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Scale numeric
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    data[num_cols] = scaler.fit_transform(data[num_cols])

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    return X, y, encoders, scaler


def preprocess_single_customer(customer_dict, encoders, scaler):
    """Preprocess a single customer for prediction"""

    customer = pd.DataFrame([customer_dict])

    # Encode categorical
    for col, le in encoders.items():
        customer[col] = le.transform(customer[col])

    # Scale numeric
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    customer[num_cols] = scaler.transform(customer[num_cols])

    return customer
