import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, mean_squared_log_error,
                             r2_score)


def evaluate_regression(y_true, y_pred, use_log = False):
    """Evaluate regression model performance.

    Args:
        y_true (array): Ground truth values.
        y_pred (array): Predicted values.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'r2': r2_score(y_true, y_pred)
    }
    if use_log:
        metrics['msle'] = mean_squared_log_error(y_true, y_pred)
    return metrics

def evaluate_classification(y_true, y_pred, labels = None):
    """
    Evaluate classification model performance.

    Args:
        y_true (array): Ground truth values.
        y_pred (array): Predicted values.
        labels (list): List of labels to index the matrix.

    Returns:
        accuracy (float): Accuracy score.
        report (pd.DataFrame): Classification report.
        confusion_matrix (pd.DataFrame): Confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred), labels, labels)
    return accuracy, report, confusion