import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_single_sample

# ------------------------------
# Load models and preprocessing objects
# ------------------------------
print("Loading models and preprocessing objects...")
dt_model = pickle.load(open("results/dt_model.pkl", "rb"))
nn_model = load_model("results/nn_model.keras")
scaler = pickle.load(open("results/scaler.pkl", "rb"))
encoders = pickle.load(open("results/encoders.pkl", "rb"))
feature_names = pickle.load(open("results/feature_names.pkl", "rb"))


# ------------------------------
# Input validation functions
# ------------------------------
def get_valid_input(prompt, valid_options, cast_type=str):
    while True:
        try:
            user_input = cast_type(input(f"{prompt}: ").strip())
            if user_input not in valid_options:
                raise ValueError
            return user_input
        except ValueError:
            print(f"Input error: Invalid input. Expected: {valid_options}")


def get_numeric_input(prompt, min_val=None, max_val=None, cast_type=float):
    while True:
        try:
            user_input = cast_type(input(f"{prompt}: ").strip())
            if (min_val is not None and user_input < min_val) or (max_val is not None and user_input > max_val):
                raise ValueError
            return user_input
        except ValueError:
            range_info = ""
            if min_val is not None and max_val is not None:
                range_info = f" between {min_val} and {max_val}"
            elif min_val is not None:
                range_info = f" greater than or equal to {min_val}"
            elif max_val is not None:
                range_info = f" less than or equal to {max_val}"
            print(f"Input error: Please enter a valid number{range_info}.")


# ------------------------------
# Main loop to collect input and predict
# ------------------------------
def main():
    print("\nPlease enter new customer details:\n")

    while True:
        try:
            # --- Categorical inputs ---
            gender = get_valid_input("gender ['Female', 'Male']", ['Female', 'Male'])
            Partner = get_valid_input("Partner ['No', 'Yes']", ['No', 'Yes'])
            Dependents = get_valid_input("Dependents ['No', 'Yes']", ['No', 'Yes'])
            PhoneService = get_valid_input("PhoneService ['No', 'Yes']", ['No', 'Yes'])
            MultipleLines = get_valid_input("MultipleLines ['No', 'No Phone Service', 'Yes']",
                                            ['No', 'No Phone Service', 'Yes'])
            InternetService = get_valid_input("InternetService ['DSL', 'Fiber Optic', 'No']",
                                              ['DSL', 'Fiber Optic', 'No'])
            OnlineSecurity = get_valid_input("OnlineSecurity ['No', 'No Internet Service', 'Yes']",
                                             ['No', 'No Internet Service', 'Yes'])
            OnlineBackup = get_valid_input("OnlineBackup ['No', 'No Internet Service', 'Yes']",
                                           ['No', 'No Internet Service', 'Yes'])
            DeviceProtection = get_valid_input("DeviceProtection ['No', 'No Internet Service', 'Yes']",
                                               ['No', 'No Internet Service', 'Yes'])
            TechSupport = get_valid_input("TechSupport ['No', 'No Internet Service', 'Yes']",
                                          ['No', 'No Internet Service', 'Yes'])
            StreamingTV = get_valid_input("StreamingTV ['No', 'No Internet Service', 'Yes']",
                                          ['No', 'No Internet Service', 'Yes'])
            StreamingMovies = get_valid_input("StreamingMovies ['No', 'No Internet Service', 'Yes']",
                                              ['No', 'No Internet Service', 'Yes'])
            Contract = get_valid_input("Contract ['Month-To-Month', 'One Year', 'Two Year']",
                                       ['Month-To-Month', 'One Year', 'Two Year'])
            PaperlessBilling = get_valid_input("PaperlessBilling ['No', 'Yes']", ['No', 'Yes'])
            PaymentMethod = get_valid_input(
                "PaymentMethod ['Bank Transfer (Automatic)', 'Credit Card (Automatic)', 'Electronic Check', 'Mailed Check']",
                ['Bank Transfer (Automatic)', 'Credit Card (Automatic)', 'Electronic Check', 'Mailed Check'])

            # --- Numeric inputs ---
            SeniorCitizen = get_valid_input("SeniorCitizen (0 = No, 1 = Yes)", [0, 1], cast_type=int)
            tenure = get_numeric_input("tenure", min_val=0, cast_type=int)
            MonthlyCharges = get_numeric_input("MonthlyCharges", min_val=0)
            TotalCharges = get_numeric_input("TotalCharges", min_val=0)

            # --- Build dataframe for preprocessing ---
            user_data = pd.DataFrame([{
                "gender": gender,
                "Partner": Partner,
                "Dependents": Dependents,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "SeniorCitizen": SeniorCitizen,
                "tenure": tenure,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges
            }])

            # --- Preprocess ---
            X = preprocess_single_sample(user_data, scaler, encoders, feature_names)

            # --- Predict ---
            dt_pred = dt_model.predict(X)[0]
            nn_pred = nn_model.predict(X)
            nn_pred_class = (nn_pred > 0.5).astype(int)[0][0]

            print("\n--- Prediction Results ---")
            print(f"Decision Tree predicts churn: {'Yes' if dt_pred == 1 else 'No'}")
            print(f"Neural Network predicts churn: {'Yes' if nn_pred_class == 1 else 'No'}")

            break  # exit loop after successful prediction

        except Exception as e:
            print(f"Input error: {e}")
            print("Please enter the details again.\n")


if __name__ == "__main__":
    main()
