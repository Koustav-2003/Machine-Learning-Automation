import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    BisectingKMeans,
)

from sklearn.metrics import silhouette_score


def _safe_silhouette_score(X, labels):
    """
    Calculate silhouette score safely.

    Silhouette score requires at least 2 clusters
    and fewer clusters than the number of samples.
    """

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return None

    if len(unique_labels) >= len(X):
        return None

    return silhouette_score(
        X,
        labels,
    )


def run_clustering(train, test):
    """
    Train and evaluate clustering models.

    Workflow:

        1. Find the best number of clusters using TRAIN data only.
        2. Train all clustering algorithms on TRAIN data.
        3. Select the best algorithm using TRAIN silhouette score.
        4. Evaluate the selected model on TEST data where possible.
        5. Return the fitted winning model.

    Models:
        - K-Means
        - Agglomerative Clustering
        - Bisecting K-Means

    Important:
        TEST data is NEVER used to select the best model
        or the best number of clusters.
    """

    # =========================================================
    # DATA
    # =========================================================

    X_train = train.copy()
    X_test = test.copy()

    # =========================================================
    # FIND BEST K USING TRAIN DATA ONLY
    # =========================================================

    # We cannot test k values greater than the number
    # of training samples minus one.
    max_k = min(
        10,
        len(X_train) - 1,
    )

    if max_k < 2:
        raise ValueError(
            "Clustering requires at least 3 training rows."
        )

    k_range = range(
        2,
        max_k + 1,
    )

    k_scores = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=0,
        )

        labels = model.fit_predict(
            X_train
        )

        score = _safe_silhouette_score(
            X_train,
            labels,
        )

        if score is not None:
            k_scores.append(score)
        else:
            k_scores.append(-np.inf)

    # =========================================================
    # SELECT BEST K
    # =========================================================

    best_k = list(k_range)[
        int(np.argmax(k_scores))
    ]

    # =========================================================
    # MODELS
    # =========================================================

    models = {

        "K-Means": KMeans(
            n_clusters=best_k,
            n_init=10,
            random_state=0,
        ),

        "Hierarchical (Agglomerative)": AgglomerativeClustering(
            n_clusters=best_k,
            linkage="ward",
        ),

        "Bisecting K-Means (Divisive)": BisectingKMeans(
            n_clusters=best_k,
            random_state=0,
        ),
    }

    # =========================================================
    # TRAIN MODELS
    # =========================================================

    results = []

    best_model = None
    best_model_name = None

    # IMPORTANT:
    # The winner is selected using TRAIN silhouette only.
    best_train_score = -np.inf

    best_test_labels = None
    best_test_score = None

    for name, model in models.items():

        # =====================================================
        # FIT ON TRAIN
        # =====================================================

        train_labels = model.fit_predict(
            X_train
        )

        # =====================================================
        # TRAIN SILHOUETTE
        # =====================================================

        train_score = _safe_silhouette_score(
            X_train,
            train_labels,
        )

        if train_score is None:
            train_score = -np.inf

        # =====================================================
        # TEST EVALUATION
        # =====================================================

        if hasattr(model, "predict"):

            # K-Means / Bisecting K-Means
            test_labels = model.predict(
                X_test
            )

            test_score = _safe_silhouette_score(
                X_test,
                test_labels,
            )

        else:

            # -------------------------------------------------
            # Agglomerative clustering does not have predict().
            #
            # We therefore cannot assign unseen test rows to
            # the clusters learned from training.
            # -------------------------------------------------

            test_labels = None
            test_score = None

        # =====================================================
        # SAVE RESULTS
        # =====================================================

        results.append(
            [
                name,
                train_score,
                test_score,
            ]
        )

        # =====================================================
        # SELECT BEST MODEL USING TRAIN ONLY
        # =====================================================

        if train_score > best_train_score:

            best_train_score = train_score

            best_model = model

            best_model_name = name

            best_test_labels = test_labels

            best_test_score = test_score

    # =========================================================
    # CHECK WINNER
    # =========================================================

    if best_model is None:

        raise ValueError(
            "Unable to select a valid clustering model."
        )

    # =========================================================
    # RESULTS DATAFRAME
    # =========================================================

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Train Silhouette Score",
            "Test Silhouette Score",
        ],
    )

    # Replace internal -inf with None
    # so it can be returned as JSON.
    results_df = results_df.replace(
        [-np.inf, np.inf],
        np.nan,
    )

    # Sort according to TRAIN performance.
    results_df = results_df.sort_values(
        "Train Silhouette Score",
        ascending=False,
    ).reset_index(drop=True)

    # =========================================================
    # RETURN
    # =========================================================

    return {

        # Actual fitted winning model.
        "best_model": best_model,

        # Name of winning algorithm.
        "best_model_name": best_model_name,

        # Best number of clusters.
        "best_k": best_k,

        # Training score used to select winner.
        "best_train_score": best_train_score,

        # Test labels if model supports prediction.
        "best_test_labels": best_test_labels,

        # Test score for winning model.
        "best_test_score": best_test_score,

        # Complete comparison.
        "results": results_df,
    }