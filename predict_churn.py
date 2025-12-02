import pandas as pd
import pickle
import tensorflow as tf
from src.preprocessing import preprocess_single_sample

# Load saved models and preprocessing objects
print("Loading models and preprocessing objects...")
dt_model = pickle.load(open("results/dt_model.pkl", "rb"))
nn_model = tf.keras.models.load_model("results/nn_model.keras")
scaler = pickle.load(open("results/scaler.pkl", "rb"))
encoders = pickle.load(open("results/encoders.pkl", "rb"))
feature_names = pickle.load(open("results/feature_names.pkl", "rb"))

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Define expected categories for validation
expected_inputs = {
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
    'PaymentMethod': ['Bank Transfer (Automatic)', 'Credit Card (Automatic)', 'Electronic Check', 'Mailed Check'],
    'SeniorCitizen': [0, 1]
}

def get_user_input():
    data = {}
    for feature, options in expected_inputs.items():
        while True:
            value = input(f"{feature} {options}: ")
            if feature == 'SeniorCitizen':
                try:
                    value = int(value)
                except:
                    print(f"Invalid input. Please enter 0 or 1")
                    continue
            if value not in options:
                print(f"Input error: Invalid input for {feature}. Expected: {options}")
            else:
                data[feature] = value
                break
    for num in numeric_cols:
        while True:
            value = input(f"{num}: ")
            try:
                data[num] = float(value)
                break
            except:
                print(f"Invalid input. Please enter a numeric value for {num}")
    return pd.DataFrame([data])

def main():
    while True:
        print("\nPlease enter new customer details:\n")
        user_df = get_user_input()
        try:
            X = preprocess_single_sample(user_df, scaler, encoders, feature_names, numeric_cols)
            dt_pred = dt_model.predict(X)[0]
            nn_pred_prob = nn_model.predict(X)[0][0]
            nn_pred = int(nn_pred_prob > 0.5)
            churn_map = {0: "No", 1: "Yes"}
            print("\nPrediction Results:")
            print(f"Decision Tree: {churn_map[dt_pred]}")
            print(f"Neural Network: {churn_map[nn_pred]} (probability: {nn_pred_prob:.2f})")
            break
        except ValueError as e:
            print(f"\n{e}\nPlease enter the details again.")

if __name__ == "__main__":
    main()
