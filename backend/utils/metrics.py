import numpy as np


def percentage(value):
    """Convert a decimal metric to a percentage."""
    if value is None:
        return None

    if not np.isfinite(value):
        return None

    return round(float(value) * 100, 2)


def format_classification_metrics(
    accuracy,
    balanced_accuracy,
    precision,
    recall,
    f1,
    roc_auc=None,
):
    """
    Convert classification metrics to percentage values
    for the API/frontend response.
    """

    return {
        "accuracy": percentage(accuracy),
        "balanced_accuracy": percentage(balanced_accuracy),
        "precision": percentage(precision),
        "recall": percentage(recall),
        "f1_score": percentage(f1),
        "roc_auc": percentage(roc_auc),
    }


def format_regression_metrics(r2, mae, rmse):
    """
    Format regression metrics.

    R2 is converted to a percentage.
    MAE and RMSE remain in their original units because
    converting them to percentages would be misleading.
    """

    return {
        "r2_score": percentage(r2),
        "mae": None if mae is None else round(float(mae), 4),
        "rmse": None if rmse is None else round(float(rmse), 4),
    }


def format_clustering_metrics(
    silhouette,
    davies_bouldin=None,
    calinski_harabasz=None,
):
    """
    Format clustering metrics.

    Silhouette Score is converted to a percentage.
    Davies-Bouldin and Calinski-Harabasz remain unchanged
    because they are not percentage metrics.
    """

    return {
        "silhouette_score": percentage(silhouette),
        "davies_bouldin_score": (
            None
            if davies_bouldin is None
            else round(float(davies_bouldin), 4)
        ),
        "calinski_harabasz_score": (
            None
            if calinski_harabasz is None
            else round(float(calinski_harabasz), 4)
        ),
    }
