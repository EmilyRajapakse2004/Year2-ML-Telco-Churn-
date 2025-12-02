import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_training_data(df, numeric_cols=None):
    """
    Preprocess full training dataset.
    Returns: X (features), y (target), scaler, encoders, feature_names
    """
    if numeric_cols is None:
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    df = df.copy()

    # Fill missing TotalCharges (if any)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols.remove('customerID')
    categorical_cols.remove('Churn')

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    df['Churn'] = target_le.fit_transform(df['Churn'].astype(str))
    encoders['Churn'] = target_le

    # Scale numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Drop customerID
    df = df.drop('customerID', axis=1, errors='ignore')

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    feature_names = X.columns.tolist()

    return X, y, scaler, encoders, feature_names


def preprocess_single_sample(df, scaler, encoders, feature_names, numeric_cols=None):
    """
    Preprocess a single customer DataFrame to match training features.
    """
    if numeric_cols is None:
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    df = df.copy()

    # Encode categorical columns using saved LabelEncoders
    for col, le in encoders.items():
        if col == 'Churn':
            continue
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError as e:
                raise ValueError(f"Input error: invalid category for {col}. Allowed: {list(le.classes_)}")

    # Scale numeric columns
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required numeric column: {col}")
    df[numeric_cols] = scaler.transform(df[numeric_cols].astype(float))

    # Add missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[feature_names]

    return df
