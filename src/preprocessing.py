# src/preprocessing.py
import pandas as pd

def preprocess_single_sample(df, scaler, encoders, feature_names, numeric_cols=['tenure','MonthlyCharges','TotalCharges']):
    """
    Preprocess single customer input for prediction.

    df: single-row DataFrame
    scaler: fitted scaler object for numeric features
    encoders: dict of LabelEncoders for categorical columns
    feature_names: list of final columns used during training
    """
    df = df.copy()

    # Validate all categorical columns
    for col, le in encoders.items():
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        val = df[col].values[0]
        if val not in le.classes_:
            raise ValueError(
                f"Column '{col}' has invalid input '{val}'. "
                f"Expected values: {list(le.classes_)}"
            )
        df[col] = le.transform(df[col].astype(str))

    # Fill missing numeric columns if any
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Missing numeric column: {col}")

    # Scale numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Reindex to match training features
    df = df.reindex(columns=feature_names, fill_value=0)

    return df
