import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_single_customer

# Load models and preprocessing tools
dt_model = joblib.load('results/dt_model.pkl')
scaler = joblib.load('results/scaler.pkl')
encoders = joblib.load('results/encoders.pkl')
nn_model = load_model('results/nn_model.keras')

# New customer info
new_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 75.5,
    'TotalCharges': 900.0
}

# Convert to DataFrame
new_customer_df = pd.DataFrame([new_customer])

# Preprocess
X_new = preprocess_single_customer(new_customer_df, scaler, encoders)

# Predictions
dt_prediction = dt_model.predict(X_new)[0]
nn_prediction = (nn_model.predict(X_new) > 0.5).astype(int)[0][0]

# Map 0/1 to No/Yes
prediction_map = {0: "No", 1: "Yes"}
print("Decision Tree Prediction:", prediction_map[dt_prediction])
print("Neural Network Prediction:", prediction_map[nn_prediction])
