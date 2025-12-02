# main.py

import pandas as pd
from src.preprocessing import preprocess_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
import pickle

# Load dataset
data = pd.read_csv("data/Telco-Customer-Churn.csv")

# Preprocess dataset
X, y, encoders, scaler = preprocess_dataset(data)

# Save encoders & scaler
pickle.dump(encoders, open("results/label_encoders.pkl", "wb"))
pickle.dump(scaler, open("results/scaler.pkl", "wb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData preprocessing done. Training models...\n")

# -------- Decision Tree --------
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

pickle.dump(dt, open("results/dt_model.pkl", "wb"))


# ---------- Neural Network ----------
nn = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

nn.save("results/nn_model.keras")

print("\nTraining completed. Models saved in results/ folder.")
