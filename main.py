import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing import preprocess_training_data
from src.utils import save_pickle

DATA_PATH = "data/Telco-Customer-Churn.csv"
RESULTS_PATH = "results/"

def build_neural_network(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    df = pd.read_csv(DATA_PATH)
    X, y, scaler, encoders, feature_names = preprocess_training_data(df)

    # Save preprocessing objects
    save_pickle(encoders, RESULTS_PATH + "encoders.pkl")
    save_pickle(scaler, RESULTS_PATH + "scaler.pkl")
    save_pickle(feature_names, RESULTS_PATH + "feature_names.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=6, criterion="gini")
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    print("\n===== DECISION TREE REPORT =====")
    print(classification_report(y_test, y_pred_dt))

    save_pickle(dt, RESULTS_PATH + "dt_model.pkl")

    # Neural Network
    nn = build_neural_network(X.shape[1])
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    nn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])
    nn.save(RESULTS_PATH + "nn_model.keras")

    y_pred_nn = (nn.predict(X_test) > 0.5).astype(int)
    print("\n===== NEURAL NETWORK REPORT =====")
    print(classification_report(y_test, y_pred_nn))

    # Save metrics summary
    metrics = {
        "DecisionTree_Accuracy": accuracy_score(y_test, y_pred_dt),
        "NeuralNet_Accuracy": accuracy_score(y_test, y_pred_nn)
    }
    save_pickle(metrics, RESULTS_PATH + "metrics_summary.pkl")

    print("\nTraining complete. All files saved in /results")

if __name__ == "__main__":
    main()
