import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test, y_test, model_type="dt"):
    """
    Evaluate the model: prints classification report,
    plots confusion matrix, and computes ROC-AUC.
    """
    if model_type == "dt":
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # If the model is a Keras model, predict probabilities
        if isinstance(model, str):
            import tensorflow as tf
            model = tf.keras.models.load_model(model)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        proba = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_type.upper()} Confusion Matrix")
    plt.show()

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, proba)
    print(f"{model_type.upper()} ROC-AUC: {roc_auc:.3f}")

    return y_pred, roc_auc
