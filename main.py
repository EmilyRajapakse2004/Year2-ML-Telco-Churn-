import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras

from src.preprocessing import preprocess_dataset
from src.utils import save_pickle

DATA_PATH = "data/Telco-Customer-Churn.csv"
RESULTS_DIR = "results/"


def build_neural_network(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Preprocessing...")
    X, y, scaler, encoders, feature_names = preprocess_dataset(df)

    # Save preprocessing objects
    save_pickle(scaler, RESULTS_DIR + "scaler.pkl")
    save_pickle(encoders, RESULTS_DIR + "encoders.pkl")
    save_pickle(feature_names, RESULTS_DIR + "feature_names.pkl")

    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============= Decision Tree =============
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=6, criterion="gini", random_state=42)
    dt.fit(X_train, y_train)

    dt_preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)

    print("\nDecision Tree Results:")
    print(classification_report(y_test, dt_preds))
    save_pickle(dt, RESULTS_DIR + "dt_model.pkl")

    # ============= Neural Network =============
    print("\nTraining Neural Network...")
    nn = build_neural_network(input_dim=X_train.shape[1])

    nn.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    nn.save(RESULTS_DIR + "nn_model.keras")

    nn_preds = (nn.predict(X_test) > 0.5).astype(int)
    nn_acc = accuracy_score(y_test, nn_preds)

    print("\nNeural Network Results:")
    print(classification_report(y_test, nn_preds))

    # Save metrics summary
    metrics = {
        "Decision Tree Accuracy": dt_acc,
        "Neural Network Accuracy": nn_acc
    }
    save_pickle(metrics, RESULTS_DIR + "metrics_summary.pkl")

    print("\nTraining complete. Models saved in /results/")


if __name__ == "__main__":
    main()
