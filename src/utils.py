# Optional helper functions for EDA or model evaluation

def summarize_metrics(y_true, y_pred):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    return {"classification_report": report, "confusion_matrix": cm.tolist(), "roc_auc": roc}
