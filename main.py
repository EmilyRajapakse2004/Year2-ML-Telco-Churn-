import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.preprocessing import preprocess_data

# Load dataset
data = pd.read_csv("data/Telco-Customer-Churn.csv")

# Preprocess
X, y, encoders, scaler, feature_names = preprocess_data(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Decision Tree =====
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("===== Decision Tree Evaluation =====")
print(classification_report(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_dt))

# Save DT model
with open("results/dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

# ===== Neural Network =====
nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Evaluate NN
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
print("===== Neural Network Evaluation =====")
print(classification_report(y_test, y_pred_nn))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_nn))

# Save NN model
nn_model.save("results/nn_model.keras")

# Save scaler, encoders, feature_names
with open("results/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("results/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("results/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
