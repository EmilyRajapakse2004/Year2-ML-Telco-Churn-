import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_single_sample

# Load trained models and preprocessing objects
print("Loading models and preprocessing objects...")
dt_model = pickle.load(open("results/dt_model.pkl", "rb"))
nn_model = load_model("results/nn_model.keras")
scaler = pickle.load(open("results/scaler.pkl", "rb"))
encoders = pickle.load(open("results/encoders.pkl", "rb"))
feature_names = pickle.load(open("results/feature_names.pkl", "rb"))

# Define expected inputs
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
]

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

valid_values = {
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
    user_data = {}
    for col in categorical_cols:
        while True:
            value = input(f"{col} {valid_values[col]}: ")
            # Convert to correct type for SeniorCitizen
            if col == 'SeniorCitizen':
                try:
                    value = int(value)
                except:
                    print("Invalid input. Enter 0 or 1.")
                    continue
            if value not in valid_values[col]:
                print(f"Invalid input for {col}. Expected: {valid_values[col]}")
            else:
                user_data[col] = value
                break
    for col in numeric_cols:
        while True:
            try:
                value = float(input(f"{col}: "))
                user_data[col] = value
                break
            except:
                print(f"Invalid input for {col}. Enter a numeric value.")
    return pd.DataFrame([user_data])

def main():
    print("Please enter new customer details:\n")
    while True:
        try:
            user_df = get_user_input()
            X = preprocess_single_sample(user_df, scaler, encoders, feature_names, numeric_cols=numeric_cols)
            # Predict with Decision Tree
            dt_pred = dt_model.predict(X)[0]
            print(f"\nDecision Tree Prediction: {'Churn' if dt_pred==1 else 'No Churn'}")
            # Predict with Neural Network
            nn_pred = nn_model.predict(X)[0][0]
            print(f"Neural Network Prediction: {'Churn' if nn_pred>=0.5 else 'No Churn'}")
            break
        except ValueError as e:
            print(f"Input error: {e}\nPlease enter the details again.\n")

if __name__ == "__main__":
    main()
