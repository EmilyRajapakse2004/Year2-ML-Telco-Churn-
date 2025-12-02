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

    # Check all required columns are present
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Encode categorical features
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError as e:
                raise ValueError(
                    f"Column '{col}' has invalid input '{df[col].values[0]}'. "
                    f"Expected values: {list(le.classes_)}"
                ) from e

    # Scale numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Reindex to match training features
    df = df.reindex(columns=feature_names, fill_value=0)

    return df
