import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_single_sample

RESULTS_PATH = "results/"

def load_pickle(name):
    return pickle.load(open(name, "rb"))

def main():
    dt_model = load_pickle(RESULTS_PATH + "dt_model.pkl")
    encoders = load_pickle(RESULTS_PATH + "encoders.pkl")
    scaler = load_pickle(RESULTS_PATH + "scaler.pkl")
    feature_names = load_pickle(RESULTS_PATH + "feature_names.pkl")
    nn_model = load_model(RESULTS_PATH + "nn_model.keras")

    print("\nEnter customer details:")
    customer = {
        "gender": input("Gender (Male/Female): "),
        "SeniorCitizen": input("SeniorCitizen (0/1): "),
        "Partner": input("Partner (Yes/No): "),
        "Dependents": input("Dependents (Yes/No): "),
        "tenure": input("Tenure (months): "),
        "PhoneService": input("PhoneService (Yes/No): "),
        "InternetService": input("InternetService (DSL/Fiber optic/No): "),
        "Contract": input("Contract (Month-to-month/One year/Two year): "),
        "PaymentMethod": input("PaymentMethod: "),
        "MonthlyCharges": input("MonthlyCharges: ")
    }

    user_df = pd.DataFrame([customer])
    X = preprocess_single_sample(user_df, scaler, encoders, feature_names)

    dt_pred = dt_model.predict(X)[0]
    nn_pred = (nn_model.predict(X)[0][0] > 0.5)

    print("\n============================")
    print(f"Decision Tree Prediction: {'Churn' if dt_pred else 'No Churn'}")
    print(f"Neural Network Prediction: {'Churn' if nn_pred else 'No Churn'}")
    print("============================\n")

if __name__ == "__main__":
    main()
