import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(df):
    """
    Preprocess the full dataset for training
    """
    df = df.copy()

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.remove('customerID') if 'customerID' in cat_cols else None

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split features and target
    X = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    y = df['Churn']

    return X, y, encoders

def preprocess_single_customer(df, scaler, encoders):
    """
    Preprocess a single customer DataFrame for prediction.
    """
    df = df.copy()

    # Handle numeric features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical features using trained encoders
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            df[col] = le.transform(df[col])
        else:
            df[col] = df[col].astype('category').cat.codes

    # Scale numeric features
    df[numeric_features] = scaler.transform(df[numeric_features])

    return df.values
