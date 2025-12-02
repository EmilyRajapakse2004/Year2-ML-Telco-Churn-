# predict_churn.py
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_single_sample

def load_objects():
    with open("results/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("results/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("results/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    dt_model = pickle.load(open("results/dt_model.pkl", "rb"))
    nn_model = load_model("results/nn_model.keras")
    return encoders, scaler, feature_names, dt_model, nn_model

def get_user_input():
    """
    Prompt the user for all features.
    """
    user_data = {}
    print("Please enter new customer details:\n")

    user_data['gender'] = input("gender ('Female', 'Male'): ").strip()
    user_data['Partner'] = input("Partner ('No', 'Yes'): ").strip()
    user_data['Dependents'] = input("Dependents ('No', 'Yes'): ").strip()
    user_data['PhoneService'] = input("PhoneService ('No', 'Yes'): ").strip()
    user_data['MultipleLines'] = input("MultipleLines ('No', 'No Phone Service', 'Yes'): ").strip()
    user_data['InternetService'] = input("InternetService ('DSL', 'Fiber Optic', 'No'): ").strip()
    user_data['OnlineSecurity'] = input("OnlineSecurity ('No', 'No Internet Service', 'Yes'): ").strip()
    user_data['OnlineBackup'] = input("OnlineBackup ('No', 'No Internet Service', 'Yes'): ").strip()
    user_data['DeviceProtection'] = input("DeviceProtection ('No', 'No Internet Service', 'Yes'): ").strip()
    user_data['TechSupport'] = input("TechSupport ('No', 'No Internet Service', 'Yes'): ").strip()
    user_data['StreamingTV'] = input("StreamingTV ('No', 'No Internet Service', 'Yes'): ").strip()
    user_data['StreamingMovies'] = input("StreamingMovies ('No', 'No Internet Service', 'Yes'): ").strip()
    user_data['Contract'] = input("Contract ('Month-To-Month', 'One Year', 'Two Year'): ").strip()
    user_data['PaperlessBilling'] = input("PaperlessBilling ('No', 'Yes'): ").strip()
    user_data['PaymentMethod'] = input("PaymentMethod ('Bank Transfer (Automatic)', 'Credit Card (Automatic)', 'Electronic Check', 'Mailed Check'): ").strip()
    user_data['SeniorCitizen'] = int(input("SeniorCitizen (0 = No, 1 = Yes): "))
    user_data['tenure'] = float(input("tenure: "))
    user_data['MonthlyCharges'] = float(input("MonthlyCharges: "))
    user_data['TotalCharges'] = float(input("TotalCharges: "))

    return pd.DataFrame([user_data])

def main():
    encoders, scaler, feature_names, dt_model, nn_model = load_objects()

    while True:
        try:
            user_df = get_user_input()
            X = preprocess_single_sample(user_df, scaler, encoders, feature_names)
            break
        except ValueError as e:
            print("Input error:", e)
            print("Please enter the details again.\n")

    # Predict
    dt_pred = dt_model.predict(X)[0]
    nn_pred = nn_model.predict(X)[0][0]

    print("\nPredictions:")
    print("Decision Tree: ", "Yes" if dt_pred == 1 else "No")
    print("Neural Network: ", "Yes" if nn_pred >= 0.5 else "No")

if __name__ == "__main__":
    main()
