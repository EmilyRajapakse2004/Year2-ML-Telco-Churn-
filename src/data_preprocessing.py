import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_clean_data(filepath):
    """
    Load the Telco Customer Churn dataset and clean it.
    """
    data = pd.read_csv(filepath)

    # Convert TotalCharges to numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    return data


def preprocess_data(data):
    """
    Encode categorical variables, scale numerical features,
    and split data into train/test sets.
    Returns X_train, X_test, y_train, y_test, scaler, num_cols
    """
    # Encode target
    le = LabelEncoder()
    data['Churn'] = le.fit_transform(data['Churn'])

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include='object').columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Split features & target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, num_cols
