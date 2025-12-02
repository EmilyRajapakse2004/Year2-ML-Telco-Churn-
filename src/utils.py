# src/utils.py

import pandas as pd
import numpy as np


def load_dataset(path):
    """Loads the Telco Churn dataset."""
    data = pd.read_csv(path)

    # Convert TotalCharges to numeric, fix errors
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    return data


def show_basic_info(df):
    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Class Distribution ---")
    print(df["Churn"].value_counts())
