from src.preprocessing import load_data, clean_data, preprocess_data
from src.models import train_decision_tree, build_neural_network
from src.evaluation import evaluate_model, plot_training_history
from sklearn.model_selection import train_test_split
import joblib

# 1️⃣ Load and clean data
data = load_data('data/Telco-Customer-Churn.csv')
data = clean_data(data)

# 2️⃣ Preprocess data
X_scaled, y, scaler = preprocess_data(data)

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4️⃣ Train Decision Tree
dt_model = train_decision_tree(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
evaluate_model(y_test, y_pred_dt, model_name="Decision Tree")

# 5️⃣ Train Neural Network
from tensorflow.keras.callbacks import EarlyStopping
nn_model = build_neural_network(X_train.shape[1])
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32,
                       validation_split=0.2, verbose=1,
                       callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
evaluate_model(y_test, y_pred_nn, model_name="Neural Network")
plot_training_history(history)

# 6️⃣ Save models
joblib.dump(dt_model, 'results/dt_model.pkl')
joblib.dump(scaler, 'results/scaler.pkl')
nn_model.save('results/nn_model.keras')
