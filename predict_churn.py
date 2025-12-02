# predict_churn.py

import pickle
import pandas as pd
from tensorflow import keras
from src.preprocessing import preprocess_single_customer

# Load models
dt_model = pickle.load(open("results/dt_model.pkl", "rb"))
nn_model = keras.models.load_model("results/nn_model.keras")

# Load encoders & scaler
encoders = pickle.load(open("results/label_encoders.pkl", "rb"))
scaler = pickle.load(open("results/scaler.pkl", "rb"))

# Example new customer input
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 250.00
}

# Preprocess new customer
processed = preprocess_single_customer(customer, encoders, scaler)

# Predictions
dt_pred = dt_model.predict(processed)[0]
nn_pred = nn_model.predict(processed)[0][0]

print("\n--- CHURN PREDICTIONS ---")
print("Decision Tree Prediction:", "Churn" if dt_pred == 1 else "Not Churn")
print("Neural Network Prediction:", "Churn" if nn_pred > 0.5 else "Not Churn")
