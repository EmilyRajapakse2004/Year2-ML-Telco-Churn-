import pandas as pd
import numpy as np


def preprocess_single_sample(df, scaler, encoders, feature_names, numeric_cols):
    """
    Preprocess a single-row DataFrame for prediction.

    df: single-row DataFrame
    scaler: fitted scaler for numeric features
    encoders: dict of LabelEncoders for categorical features
    feature_names: list of all features used in training
    numeric_cols: list of numeric features
    """
    df = df.copy()

    # 1️⃣ Validate input
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2️⃣ Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform([df[col][0]])
            except ValueError:
                raise ValueError(f"Invalid value for '{col}': {df[col][0]}. "
                                 f"Expected one of: {list(le.classes_)}")

    # 3️⃣ Scale numeric columns
    df_numeric = df[numeric_cols].astype(float)
    df_numeric_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=numeric_cols)

    # 4️⃣ Combine scaled numeric + encoded categorical in training order
    final_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in numeric_cols:
            final_df[col] = df_numeric_scaled[col]
        else:
            final_df[col] = df[col]

    return final_df.values
