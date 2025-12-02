import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_dataset(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(0, inplace=True)

    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    df = df.drop(["customerID", "Churn"], axis=1)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    feature_names = df.columns.tolist()
    X = df.values

    return X, y.values, scaler, encoders, feature_names


def preprocess_single_sample(df, scaler, encoders, feature_names):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].map(lambda x: x if x in le.classes_ else "Unknown")
            if "Unknown" not in le.classes_:
                le.classes_ = list(le.classes_) + ["Unknown"]
            df[col] = le.transform(df[col].astype(str))

    df[numeric_cols] = scaler.transform(df[numeric_cols])

    df = df.reindex(columns=feature_names, fill_value=0)

    return df.values
