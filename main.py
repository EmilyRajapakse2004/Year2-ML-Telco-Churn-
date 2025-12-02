import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.preprocessing import preprocess_dataset
import os

# Create results folder
os.makedirs('results', exist_ok=True)

# Load dataset
data = pd.read_csv('data/Telco-Customer-Churn.csv')

# Preprocess dataset
X, y, encoders = preprocess_dataset(data)

# Scale numeric features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Decision Tree ------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("===== Decision Tree Evaluation =====")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_dt))

# Save Decision Tree and scaler
joblib.dump(dt_model, 'results/dt_model.pkl')
joblib.dump(scaler, 'results/scaler.pkl')
joblib.dump(encoders, 'results/encoders.pkl')

# ------------------ Neural Network ------------------
nn_model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=2)

y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
print("===== Neural Network Evaluation =====")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_nn))

# Save Neural Network
nn_model.save('results/nn_model.keras')
