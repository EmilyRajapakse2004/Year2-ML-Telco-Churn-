from src.data_preprocessing import load_and_clean_data, preprocess_data
from src.models import train_decision_tree, train_neural_network
from src.evaluation import evaluate_model
from src.predict import predict_new_customer
import joblib
import os
import pandas as pd

# ------------------------------
# 0️⃣ Initialize CSV for new customers
# ------------------------------
NEW_CUSTOMER_CSV = "results/new_customers.csv"
os.makedirs("results", exist_ok=True)  # Ensure folder exists

# Define CSV columns
csv_columns = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen', 'tenure',
    'MonthlyCharges', 'TotalCharges', 'Prediction'
]

# Create a fresh CSV at runtime with headers
pd.DataFrame(columns=csv_columns).to_csv(NEW_CUSTOMER_CSV, index=False)
print("Temporary new customer CSV initialized:", NEW_CUSTOMER_CSV)

# ------------------------------
# 1️⃣ Load and preprocess data
# ------------------------------
data = load_and_clean_data("data/Telco-Customer-Churn.csv")
X_train, X_test, y_train, y_test, scaler, num_cols = preprocess_data(data)

# ------------------------------
# 2️⃣ Train or load models
# ------------------------------
if not os.path.exists("results/decision_tree_model.pkl"):
    dt_model = train_decision_tree(X_train, y_train)
    joblib.dump(dt_model, "results/decision_tree_model.pkl")
else:
    dt_model = joblib.load("results/decision_tree_model.pkl")

if not os.path.exists("results/neural_network_model.keras"):
    nn_model, history = train_neural_network(X_train, y_train)
    nn_model.save("results/neural_network_model.keras")
else:
    nn_model = None  # Will be loaded in predict function

# ------------------------------
# 3️⃣ Evaluate models
# ------------------------------
print("\n-------------------------------------------------------------------")
print("Evaluating Decision Tree:")
evaluate_model(dt_model, X_test, y_test, model_type="dt")

print("\n-------------------------------------------------------------------")
print("Evaluating Neural Network:")
evaluate_model(nn_model if nn_model else "results/neural_network_model.keras",
               X_test, y_test, model_type="nn")
print("\n-------------------------------------------------------------------")

# ------------------------------
# 4️⃣ Prepare input validation options
# ------------------------------
categorical_options = {
    'gender': ['Male', 'Female'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

numeric_fields = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# ------------------------------
# 5️⃣ Function to save customer to CSV
# ------------------------------
def save_new_customer(customer_dict):
    # Add prediction column if missing
    if 'Prediction' not in customer_dict:
        customer_dict['Prediction'] = ""

    # Convert to DataFrame
    df = pd.DataFrame([customer_dict])

    # Reorder columns
    df = df.reindex(columns=csv_columns)

    # Append to CSV (or create new)
    if os.path.exists(NEW_CUSTOMER_CSV) and os.path.getsize(NEW_CUSTOMER_CSV) > 0:
        df.to_csv(NEW_CUSTOMER_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(NEW_CUSTOMER_CSV, mode="w", header=True, index=False)

    print("Customer saved to CSV:", NEW_CUSTOMER_CSV)

# ------------------------------
# 6️⃣ Customer prediction loop
# ------------------------------
while True:
    print("\nEnter new customer details to predict churn:")

    # Collect categorical input with validation
    new_customer = {}
    for attr in categorical_options.keys():
        while True:
            value = input(f"{attr} {categorical_options[attr]}: ").strip()
            if value in categorical_options[attr]:
                new_customer[attr] = value
                break
            else:
                print(f"Invalid input. Please enter one of {categorical_options[attr]}")

    # Collect numeric input with validation
    for attr in numeric_fields:
        while True:
            try:
                value = float(input(f"{attr}: ").strip())
                new_customer[attr] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Choose model
    model_choice = ""
    while model_choice not in ["nn", "dt"]:
        model_choice = input("Choose model for prediction (nn/dt): ").lower()

    # Make prediction
    prediction = predict_new_customer(
        new_customer,
        X_train.columns,
        scaler,
        num_cols,
        dt_model=dt_model,
        nn_model_path="results/neural_network_model.keras",
        model_type=model_choice
    )

    print("\nPrediction for this customer:", prediction)

    # Save customer + prediction to CSV
    customer_with_pred = new_customer.copy()
    customer_with_pred['Prediction'] = prediction
    save_new_customer(customer_with_pred)

    # Ask if user wants to predict another customer
    again = input("\nDo you want to predict another customer? (yes/no): ").strip().lower()
    if again != "yes":
        print("Exiting system. Goodbye!")
        break
