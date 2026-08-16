import pandas as pd


def detect_task(target: pd.Series) -> str:
    """
    Detect whether a supervised learning problem is classification
    or regression.

    Rule:
        fewer than 10 unique target values -> classification
        10 or more unique target values -> regression
    """

    unique_values = target.nunique(dropna=True)

    if unique_values < 2:
        raise ValueError(
            "The target must contain at least 2 unique values."
        )

    if unique_values < 10:
        return "classification"

    return "regression"
