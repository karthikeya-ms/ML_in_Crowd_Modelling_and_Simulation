"""Contains any evaluation metrics related code."""

from sklearn.metrics import classification_report
import pandas as pd


def get_classification_metric_summary(y, y_pred):
    """
    Compute classification metrics summary and accuracy for a classification task.

    Parameters:
    - y (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - class_metrics (pandas.DataFrame): DataFrame containing precision, recall, f1-score, and support for each class.
    - accuracy (float): Overall accuracy of the classification.
    """
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report)
    class_metrics = (
        report_df.drop(columns=["accuracy", "macro avg"])
        .rename(columns={"weighted avg": "average"})
        .iloc[:-1]
    )
    accuracy = report["accuracy"]
    return class_metrics, accuracy
