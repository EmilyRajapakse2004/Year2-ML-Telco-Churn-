import pandas as pd
import pickle
from tensorflow import keras
from src.preprocessing import preprocess_single_sample

RESULTS_DIR = "results/"


def load_artifacts():
    dt = pickle.load(open(RESULTS_DIR + "dt_model.pkl", "rb"))
    nn = keras.models.load_model(RESULTS_DIR + "nn_model.keras")
    scaler = pickle.load(open(RESULTS_DIR + "scaler.pkl", "rb"))
    encoders = pickle.load(open(RESULTS_DIR + "encoders.pkl", "rb"))
    feature_names = pickle.load(open(RESULTS_DIR + "feature_names.pkl", "rb"))
    return dt, nn, scaler, encoders, feature_names


def ask_user():
    print("\nEnter new customer info:")
    data = {}

    data["gender"] = input("Gender (Male/Female): ")
    data["SeniorCitizen"] = int(input("Senior Citizen (0/1): "))
    data["Partner"] = input("Partner (Yes/No): ")
    data["Dependents"] = input("Dependents (Yes/No): ")
    data["tenure"] = float(input("Tenure (months): "))
    data["PhoneService"] = input("PhoneService (Yes/No): ")
    data["MultipleLines"] = input("MultipleLines (Yes/No/No phone service): ")
    data["InternetService"] = input("InternetService (DSL/Fiber optic/No): ")
    data["OnlineSecurity"] = input("OnlineSecurity (Yes/No/No internet service): ")
    data["OnlineBackup"] = input("OnlineBackup (Yes/No/No internet service): ")
    data["DeviceProtection"] = input("DeviceProtection (Yes/No/No internet service): ")
    data["TechSupport"] = input("TechSupport (Yes/No/No internet service): ")
    data["StreamingTV"] = input("StreamingTV (Yes/No/No internet service): ")
    data["StreamingMovies"] = input("StreamingMovies (Yes/No/No internet service): ")
    data["Contract"] = input("Contract (Month-to-month/One year/Two year): ")
    data["PaperlessBilling"] = input("PaperlessBilling (Yes/No): ")
    data["PaymentMethod"] = input(
        "PaymentMethod (Electronic check/Mailed check/Bank transfer/Credit card): "
    )
    data["MonthlyCharges"] = float(input("MonthlyCharges: "))
    data["TotalCharges"] = float(input("TotalCharges: "))

    return pd.DataFrame([data])


def main():
    print("Loading models...")
    dt, nn, scaler, encoders, feature_names = load_artifacts()

    user_df = ask_user()

    X = preprocess_single_sample(user_df, scaler, encoders, feature_names)

    # Predictions
    dt_pred = dt.predict(X)[0]
    nn_pred_proba = nn.predict(X)[0][0]
    nn_pred = 1 if nn_pred_proba > 0.5 else 0

    print("\n=== PREDICTION RESULTS ===")
    print(f"Decision Tree Prediction: {'Churn' if dt_pred else 'No Churn'}")
    print(f"Neural Network Prediction: {'Churn' if nn_pred else 'No Churn'}")
    print(f"Neural Network Probability: {nn_pred_proba:.4f}")


if __name__ == "__main__":
    main()
