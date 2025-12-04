# main.py
import pandas as pd
from src.data_preprocessing import load_and_clean_data, preprocess_data
from src.models import train_decision_tree, train_neural_network
from src.evaluation import evaluate_model
from src.predict import predict_new_customer
import tensorflow as tf
import os

# GLOBALS
df = None
X_train = X_test = y_train = y_test = None
scaler = None
num_cols = None
dt_model = None
nn_model = None
X_columns = None

RESULTS_CSV = "results/new_customers.csv"
os.makedirs("results", exist_ok=True)


# 1. LOAD DATASET
def load_dataset():
    global df
    try:
        df = load_and_clean_data("data/Telco-Customer-Churn.csv")
        print("✔ Dataset loaded and cleaned successfully!")
    except Exception as e:
        print(f"✘ Error loading dataset: {e}")


# 2. DISPLAY BASIC EDA

def show_eda():
    if df is None:
        print("✘ Load dataset first!")
        return

    print("=== Dataset Overview ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Churn Distribution: {df['Churn'].value_counts().to_dict()}")


# 3. PREPROCESS DATA
def preprocess():
    global X_train, X_test, y_train, y_test, scaler, num_cols, X_columns

    if df is None:
        print("✘ Load dataset first!")
        return

    X_train, X_test, y_train, y_test, scaler, num_cols = preprocess_data(df)
    X_columns = X_train.columns
    print("✔ Preprocessing completed!")


# 4. TRAIN DECISION TREE
def train_dt():
    global dt_model
    if X_train is None:
        print("✘ Preprocess data first!")
        return

    dt_model = train_decision_tree(X_train, y_train, max_depth=5)
    print("Evaluating Decision Tree...")
    evaluate_model(dt_model, X_test, y_test, model_type="dt")
    print("✔ Decision Tree Training Complete")


# 5. TRAIN NEURAL NETWORK
def train_nn():
    global nn_model
    if X_train is None:
        print("✘ Preprocess data first!")
        return

    nn_model, history = train_neural_network(X_train, y_train, epochs=20, batch_size=32)
    nn_model.save("nn_model.h5")
    print("✔ Neural Network trained and saved as nn_model.h5")
    print("Evaluating Neural Network...")
    evaluate_model(nn_model, X_test, y_test, model_type="nn")


# 6. PREDICT NEW CUSTOMER
def predict_customer():
    if scaler is None or X_columns is None:
        print("✘ Preprocess data & train at least one model first!")
        return

    print("Enter full customer details for prediction:")

    # Categorical options
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

    # Collect categorical input
    customer = {}
    for field, options in categorical_options.items():
        while True:
            value = input(f"{field} {options}: ").strip()
            if value in options:
                customer[field] = value
                break
            print(f"Invalid input. Choose one of {options}")

    # Collect numeric input
    for field in numeric_fields:
        while True:
            try:
                value = float(input(f"{field}: "))
                if field == 'SeniorCitizen':
                    value = int(value)
                customer[field] = value
                break
            except ValueError:
                print("Invalid input. Enter a number.")

    # Choose model
    print("Choose Model:")
    print("1. Neural Network")
    print("2. Decision Tree")
    while True:
        choice = input("Your choice (1/2): ").strip()
        if choice in ["1", "2"]:
            break

    if choice == "1":
        prediction = predict_new_customer(
            customer, X_columns, scaler, num_cols, nn_model_path="nn_model.h5", model_type="nn"
        )
    else:
        prediction = predict_new_customer(
            customer, X_columns, scaler, num_cols, dt_model=dt_model, model_type="dt"
        )

    print(f"Prediction: {prediction}")

    # Save to CSV
    customer['Prediction'] = prediction
    df_to_save = pd.DataFrame([customer])
    if os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0:
        df_to_save.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_to_save.to_csv(RESULTS_CSV, index=False)
    print(f"Customer saved to {RESULTS_CSV}")


# MENU
def menu():
    while True:
        print("\n========= TELCO CHURN SYSTEM =========")
        print("1. Load Dataset")
        print("2. Show EDA")
        print("3. Preprocess Data")
        print("4. Train Decision Tree")
        print("5. Train Neural Network")
        print("6. Predict New Customer")
        print("7. Exit")

        choice = input("Select an option (1-7): ").strip()
        if choice == "1":
            load_dataset()
        elif choice == "2":
            show_eda()
        elif choice == "3":
            preprocess()
        elif choice == "4":
            train_dt()
        elif choice == "5":
            train_nn()
        elif choice == "6":
            predict_customer()
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Choose 1-7.")


# RUN MAIN MENU
if __name__ == "__main__":
    menu()
