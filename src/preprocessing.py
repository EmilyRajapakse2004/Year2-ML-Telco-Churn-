import pandas as pd
import numpy as np

def preprocess_single_sample(df, scaler, encoders, feature_names):
    df = df.copy()

    # Separate categorical and numeric columns
    cat_cols = list(encoders.keys())
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    # Encode categorical columns
    for col in cat_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        le = encoders[col]
        try:
            df[col] = le.transform(df[col].astype(str))
        except ValueError as e:
            raise ValueError(
                f"Invalid value for {col}. Allowed values: {list(le.classes_)}"
            ) from e

    # Ensure numeric columns exist
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required numeric column: {col}")
        df[col] = pd.to_numeric(df[col])

    # Scale numeric columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Add any missing columns from training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[feature_names]

    return df
