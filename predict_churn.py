import pandas as pd
import pickle
from src.preprocessing import preprocess_single_sample
from tensorflow.keras.models import load_model

def get_input(prompt, allowed_values=None, dtype=str):
    while True:
        val = input(f"{prompt}: ").strip()
        try:
            val_cast = dtype(val)
            if allowed_values is not None:
                # For integers
                if val_cast not in allowed_values:
                    raise ValueError
            return val_cast
        except:
            if allowed_values is not None:
                print(f"Input error: Invalid input. Expected: {allowed_values}")
            else:
                print("Input error: Please enter a valid value.")

def main():
    print("Loading models and preprocessing objects...")
    dt_model = pickle.load(open("results/dt_model.pkl", "rb"))
    nn_model = load_model("results/nn_model.keras")
    scaler = pickle.load(open("results/scaler.pkl", "rb"))
    encoders = pickle.load(open("results/encoders.pkl", "rb"))
    feature_names = pickle.load(open("results/feature_names.pkl", "rb"))

    while True:
        print("\nPlease enter new customer details:")

        # Collect user input
        user_data = {
            'gender': get_input("gender", list(encoders['gender'].classes_)),
            'Partner': get_input("Partner", list(encoders['Partner'].classes_)),
            'Dependents': get_input("Dependents", list(encoders['Dependents'].classes_)),
            'PhoneService': get_input("PhoneService", list(encoders['PhoneService'].classes_)),
            'MultipleLines': get_input("MultipleLines", list(encoders['MultipleLines'].classes_)),
            'InternetService': get_input("InternetService", list(encoders['InternetService'].classes_)),
            'OnlineSecurity': get_input("OnlineSecurity", list(encoders['OnlineSecurity'].classes_)),
            'OnlineBackup': get_input("OnlineBackup", list(encoders['OnlineBackup'].classes_)),
            'DeviceProtection': get_input("DeviceProtection", list(encoders['DeviceProtection'].classes_)),
            'TechSupport': get_input("TechSupport", list(encoders['TechSupport'].classes_)),
            'StreamingTV': get_input("StreamingTV", list(encoders['StreamingTV'].classes_)),
            'StreamingMovies': get_input("StreamingMovies", list(encoders['StreamingMovies'].classes_)),
            'Contract': get_input("Contract", list(encoders['Contract'].classes_)),
            'PaperlessBilling': get_input("PaperlessBilling", list(encoders['PaperlessBilling'].classes_)),
            'PaymentMethod': get_input("PaymentMethod", list(encoders['PaymentMethod'].classes_)),
            'SeniorCitizen': get_input("SeniorCitizen (0=No, 1=Yes)", [0, 1], int),
            'tenure': get_input("tenure", dtype=float),
            'MonthlyCharges': get_input("MonthlyCharges", dtype=float),
            'TotalCharges': get_input("TotalCharges", dtype=float),
        }

        user_df = pd.DataFrame([user_data])

        try:
            X = preprocess_single_sample(user_df, scaler, encoders, feature_names)
        except ValueError as e:
            print(f"Input error: {e}\nPlease enter the details again.\n")
            continue

        # Predict using Decision Tree
        dt_pred = dt_model.predict(X)[0]
        dt_prob = dt_model.predict_proba(X)[0][1]

        # Predict using Neural Network
        nn_prob = nn_model.predict(X, verbose=0)[0][0]
        nn_pred = int(nn_prob >= 0.5)

        print("\nPrediction Results:")
        print(f"Decision Tree: {'Churn' if dt_pred else 'No Churn'} (Prob={dt_prob:.2f})")
        print(f"Neural Network: {'Churn' if nn_pred else 'No Churn'} (Prob={nn_prob:.2f})")

        break

if __name__ == "__main__":
    main()
