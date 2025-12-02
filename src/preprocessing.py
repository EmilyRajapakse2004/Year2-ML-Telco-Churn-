import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_training_data(df):
    df = df.copy()

    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(0, inplace=True)

    y = df["Churn"].replace({"Yes": 1, "No": 0})
    df.drop(columns=["Churn"], inplace=True)

    encoders = {}
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df.values, y.values, scaler, encoders, list(df.columns)

def preprocess_single_sample(df, scaler, encoders, feature_names):
    df = df.copy()

    # Encode categorical features
    for col, le in encoders.items():
        if col in df:
            df[col] = le.transform(df[col].astype(str))

    # Convert numeric columns safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Put in correct column order
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale numeric
    df[df.columns] = scaler.transform(df)

    return df.values
