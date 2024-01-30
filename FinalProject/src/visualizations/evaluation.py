from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(y, y_pred, title):
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")


def plot_classification_metric_summary(class_metrics, title):
    plt.figure(figsize=(16, 4))
    sns.heatmap(class_metrics, annot=True, cmap="Blues", cbar=True)
    plt.hlines(
        [1, 2], *plt.xlim(), colors=["black", "black"], linestyles=["dashed", "dashed"]
    )
    plt.title(title)
