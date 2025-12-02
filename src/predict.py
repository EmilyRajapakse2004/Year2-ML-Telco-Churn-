import pandas as pd
import tensorflow as tf

nn_model = None  # global variable to reuse the NN model


def predict_new_customer(customer_dict, X_train_columns, scaler, num_cols,
                         dt_model=None, nn_model_path=None, model_type="nn"):
    """
    Predict churn for a new customer.
    customer_dict: dict of customer attributes
    X_train_columns: columns from training data
    scaler: fitted scaler for numerical features
    dt_model: loaded Decision Tree model
    nn_model_path: path to saved Neural Network model
    model_type: "nn" or "dt"
    """
    global nn_model
    df = pd.DataFrame([customer_dict])

    # One-hot encode and align with training columns
    df = pd.get_dummies(df)
    df = df.reindex(columns=X_train_columns, fill_value=0)

    # Scale numerical features
    df[num_cols] = scaler.transform(df[num_cols])

    if model_type == "nn":
        if nn_model is None:
            nn_model = tf.keras.models.load_model(nn_model_path)
        pred = (nn_model.predict(df) > 0.5).astype("int32")[0][0]
    else:
        pred = dt_model.predict(df)[0]

    return "Churn" if pred == 1 else "Not Churn"
