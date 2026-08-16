import pandas as pd
from pandas.api.types import is_numeric_dtype


class ValidationError(Exception):
    """Raised when uploaded ML data is invalid."""
    pass


def validate_supervised_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
):
    """
    Validate train/test data for supervised learning.
    """

    if train.empty:
        raise ValidationError("Training file is empty.")

    if test.empty:
        raise ValidationError("Test file is empty.")

    if target_column not in train.columns:
        raise ValidationError(
            f"Target column '{target_column}' not found in training data."
        )

    if target_column not in test.columns:
        raise ValidationError(
            f"Target column '{target_column}' not found in test data."
        )

    if train[target_column].isnull().any():
        raise ValidationError(
            f"Target column '{target_column}' contains missing values in training data."
        )

    if test[target_column].isnull().any():
        raise ValidationError(
            f"Target column '{target_column}' contains missing values in test data."
        )

    train_features = set(train.columns) - {target_column}
    test_features = set(test.columns) - {target_column}

    if train_features != test_features:
        raise ValidationError(
            "Train and test datasets must contain the same feature columns."
        )

    return {
        "task": "supervised",
        "target_column": target_column,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_features": len(train_features),
        "n_target_values": train[target_column].nunique(),
    }


def detect_task(target: pd.Series) -> str:
    """
    Automatically determine whether a supervised target
    represents classification or regression.

    Rules:
    - Non-numeric target -> classification
    - Numeric target with relatively few unique values -> classification
    - Numeric target with many unique values -> regression
    """

    if not is_numeric_dtype(target):
        return "classification"

    # Numeric categorical targets such as 0/1, 0/1/2, etc.
    if target.nunique() <= 20:
        return "classification"

    return "regression"


def validate_classification_targets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
):
    """
    Validate target values for classification.
    """

    train_classes = set(
        train[target_column].dropna().unique()
    )

    test_classes = set(
        test[target_column].dropna().unique()
    )

    if len(train_classes) < 2:
        raise ValidationError(
            "Classification requires at least two classes in the training data."
        )

    unseen_test_classes = test_classes - train_classes

    if unseen_test_classes:
        raise ValidationError(
            "Test data contains classes that are not present "
            f"in training data: {list(unseen_test_classes)}"
        )


def validate_regression_target(
    train: pd.DataFrame,
    target_column: str,
):
    """
    Validate target values for regression.
    """

    if not is_numeric_dtype(train[target_column]):
        raise ValidationError(
            "Regression target must be numeric."
        )

    if train[target_column].isnull().any():
        raise ValidationError(
            "Regression target contains missing values."
        )


def validate_unsupervised_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Validate train/test data for unsupervised learning.
    """

    if train.empty:
        raise ValidationError(
            "Training file is empty."
        )

    if test.empty:
        raise ValidationError(
            "Test file is empty."
        )

    if list(train.columns) != list(test.columns):
        raise ValidationError(
            "Train and test datasets must contain "
            "the same feature columns in the same order."
        )

    return {
        "task": "unsupervised",
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_features": len(train.columns),
    }