# predict_churn.py
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_single_sample


def prompt_user_input():
    user_data = {}

    # Categorical columns with allowed values
    categorical_options = {
        'gender': ['Female', 'Male'],
        'Partner': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No', 'No Phone Service', 'Yes'],
        'InternetService': ['DSL', 'Fiber Optic', 'No'],
        'OnlineSecurity': ['No', 'No Internet Service', 'Yes'],
        'OnlineBackup': ['No', 'No Internet Service', 'Yes'],
        'DeviceProtection': ['No', 'No Internet Service', 'Yes'],
        'TechSupport': ['No', 'No Internet Service', 'Yes'],
        'StreamingTV': ['No', 'No Internet Service', 'Yes'],
        'StreamingMovies': ['No', 'No Internet Service', 'Yes'],
        'Contract': ['Month-To-Month', 'One Year', 'Two Year'],
        'PaperlessBilling': ['No', 'Yes'],
        'PaymentMethod': ['Bank Transfer (Automatic)', 'Credit Card (Automatic)',
                          'Electronic Check', 'Mailed Check']
    }

    # Prompt for categorical inputs
    for col, options in categorical_options.items():
        while True:
            val = input(f"{col} {tuple(options)}: ").strip()
            if val in options:
                user_data[col] = val
                break
            else:
                print(f"Invalid input. Please enter one of {options}")

    # Special case: SeniorCitizen (0 or 1)
    while True:
        val = input("SeniorCitizen (0 = No, 1 = Yes): ").strip()
        if val in ['0', '1']:
            user_data['SeniorCitizen'] = int(val)
            break
        else:
            print("Invalid input. Enter 0 (No) or 1 (Yes)")

    # Numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        while True:
            try:
                val = float(input(f"{col}: ").strip())
                user_data[col] = val
                break
            except ValueError:
                print(f"Invalid input. Enter a numeric value for {col}.")

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])
    return user_df


def main():
    print("Loading models and preprocessing objects...")
    # Load trained models and preprocessing objects
    dt_model = pickle.load(open("results/dt_model.pkl", "rb"))
    nn_model = load_model("results/nn_model.keras")
    scaler = pickle.load(open("results/scaler.pkl", "rb"))
    encoders = pickle.load(open("results/encoders.pkl", "rb"))
    feature_names = pickle.load(open("results/feature_names.pkl", "rb"))

    print("\nPlease enter new customer details:\n")
    user_df = prompt_user_input()

    try:
        # Preprocess user input
        X = preprocess_single_sample(user_df, scaler, encoders, feature_names,
                                     numeric_cols=['tenure', 'MonthlyCharges', 'TotalCharges'])
    except ValueError as e:
        print(f"Input error: {e}")
        return

    # Decision Tree prediction
    dt_pred = dt_model.predict(X)[0]
    # Neural Network prediction (assume output is probability)
    nn_prob = nn_model.predict(X)[0][0]
    nn_pred = 1 if nn_prob >= 0.5 else 0

    churn_dict = {0: "No", 1: "Yes"}
    print("\nPredictions for this customer:")
    print(f"Decision Tree model: {churn_dict[dt_pred]}")
    print(f"Neural Network model: {churn_dict[nn_pred]} (probability: {nn_prob:.2f})")


if __name__ == "__main__":
    main()
