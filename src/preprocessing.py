import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    """Load CSV dataset."""
    data = pd.read_csv(path)
    return data


def clean_data(data):
    """Clean dataset: drop ID, handle missing TotalCharges."""
    if 'customerID' in data.columns:
        data.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric, fill NaN with median
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    return data


def preprocess_data(data):
    """Encode categorical variables and scale features."""
    # Encode categorical variables
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Encode target
    if data['Churn'].dtype == 'object':
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # Separate features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
