from sklearn.metrics import classification_report
import pandas as pd


def get_classification_metric_summary(y, y_pred):
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report)
    class_metrics = (
        report_df.drop(columns=["accuracy", "macro avg"])
        .rename(columns={"weighted avg": "average"})
        .iloc[:-1]
    )
    accuracy = report["accuracy"]
    return class_metrics, accuracy
