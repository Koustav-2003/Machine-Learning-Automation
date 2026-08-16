import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_roc_auc(model, X_test, y_test):
    """
    Calculate ROC-AUC when the model supports
    probabilities or decision scores.

    Returns None when ROC-AUC cannot be calculated.
    """

    try:

        number_of_classes = len(
            np.unique(y_test)
        )

        # -------------------------------------------------
        # MODELS WITH predict_proba()
        # -------------------------------------------------

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = (
                model.predict_proba(
                    X_test
                )
            )

            if number_of_classes == 2:

                return roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )

            return roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )

        # -------------------------------------------------
        # MODELS WITHOUT predict_proba()
        # -------------------------------------------------

        if hasattr(
            model,
            "decision_function",
        ):

            decision_scores = (
                model.decision_function(
                    X_test
                )
            )

            if number_of_classes == 2:

                return roc_auc_score(
                    y_test,
                    decision_scores,
                )

            return roc_auc_score(
                y_test,
                decision_scores,
                multi_class="ovr",
                average="weighted",
            )

    except Exception:

        # ROC-AUC may not be available for
        # certain datasets/models.
        return None

    return None


def run_classification(
    train,
    test,
    target_column,
):
    """
    Train and evaluate classification models.

    Lite version:
        - 2-fold StratifiedKFold
        - RandomizedSearchCV
        - 3 random parameter combinations
        - Reduced hyperparameter ranges
        - Single CPU worker

    train.csv is used for:
        - Cross-validation
        - Hyperparameter tuning
        - Final model fitting

    test.csv is used only for:
        - Final evaluation

    Returns:
        best_model
        best_model_name
        best_predictions
        best_hyperparameters
        best_roc_auc
        results
    """

    # =========================================================
    # FEATURES / TARGET
    # =========================================================

    X = train.drop(
        columns=[target_column]
    )

    y = train[target_column]

    X_test = test.drop(
        columns=[target_column]
    )

    y_test = test[target_column]

    # =========================================================
    # LITE CROSS VALIDATION
    # =========================================================

    # 2 folds instead of 5.
    CV = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=0,
    )

    # Only 3 random parameter combinations
    # are tested for each model.
    RANDOM_SEARCH_ITERATIONS = 3

    # Use a single worker to keep resource usage
    # suitable for hosted environments.
    N_JOBS = 1

    # =========================================================
    # MODELS + HYPERPARAMETER SEARCH
    # =========================================================

    models = {

        # -----------------------------------------------------
        # Logistic Regression
        # -----------------------------------------------------

        "Logistic Regression": RandomizedSearchCV(
            LogisticRegression(
                max_iter=2000,
                random_state=0,
            ),
            {
                "C": [
                    0.1,
                    1,
                    10,
                ],
                "solver": [
                    "lbfgs",
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # KNN
        # -----------------------------------------------------

        "KNN": RandomizedSearchCV(
            KNeighborsClassifier(),
            {
                "n_neighbors": [
                    3,
                    5,
                    7,
                ],
                "weights": [
                    "uniform",
                    "distance",
                ],
                "metric": [
                    "euclidean",
                    "manhattan",
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # SVM
        # -----------------------------------------------------

        "SVM": RandomizedSearchCV(
            SVC(
                probability=True,
                random_state=0,
            ),
            {
                "C": [
                    1,
                    10,
                    100,
                ],
                "kernel": [
                    "linear",
                    "rbf",
                ],
                "gamma": [
                    "scale",
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # Decision Tree
        # -----------------------------------------------------

        "Decision Tree": RandomizedSearchCV(
            DecisionTreeClassifier(
                random_state=0,
            ),
            {
                "max_depth": [
                    None,
                    5,
                    10,
                ],
                "min_samples_split": [
                    2,
                    5,
                ],
                "min_samples_leaf": [
                    1,
                    2,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # Random Forest
        # -----------------------------------------------------

        "Random Forest": RandomizedSearchCV(
            RandomForestClassifier(
                random_state=0,
                n_jobs=N_JOBS,
            ),
            {
                "n_estimators": [
                    50,
                    100,
                ],
                "max_depth": [
                    None,
                    10,
                ],
                "min_samples_split": [
                    2,
                    5,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # XGBoost
        # -----------------------------------------------------

        "XGBoost": RandomizedSearchCV(
            XGBClassifier(
                random_state=0,
                eval_metric="logloss",
                n_jobs=N_JOBS,
            ),
            {
                "n_estimators": [
                    50,
                    100,
                ],
                "learning_rate": [
                    0.05,
                    0.1,
                ],
                "max_depth": [
                    3,
                    5,
                ],
                "subsample": [
                    0.8,
                    1.0,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # LightGBM
        # -----------------------------------------------------

        "LightGBM": RandomizedSearchCV(
            LGBMClassifier(
                random_state=0,
                verbosity=-1,
                n_jobs=N_JOBS,
            ),
            {
                "n_estimators": [
                    50,
                    100,
                ],
                "learning_rate": [
                    0.05,
                    0.1,
                ],
                "num_leaves": [
                    15,
                    31,
                ],
                "max_depth": [
                    -1,
                    10,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # CatBoost
        # -----------------------------------------------------

        "CatBoost": RandomizedSearchCV(
            CatBoostClassifier(
                verbose=False,
                random_seed=0,
                thread_count=N_JOBS,
            ),
            {
                "iterations": [
                    50,
                    100,
                ],
                "learning_rate": [
                    0.05,
                    0.1,
                ],
                "depth": [
                    4,
                    6,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="accuracy",
            random_state=0,
            n_jobs=N_JOBS,
        ),
    }

    # =========================================================
    # TRAIN + EVALUATE
    # =========================================================

    results = []

    best_model = None
    best_model_name = None
    best_predictions = None
    best_hyperparameters = None
    best_roc_auc = None

    # Best model is selected using CV accuracy.
    # The test set is NOT used to select the winner.
    best_cv_score = -np.inf

    # =========================================================
    # TRAIN EVERY MODEL
    # =========================================================

    for name, model in models.items():

        # -----------------------------------------------------
        # TRAIN
        # -----------------------------------------------------

        model.fit(
            X,
            y,
        )

        # RandomizedSearchCV refits the best estimator
        # on the complete training dataset by default.
        fitted_model = (
            model.best_estimator_
        )

        # -----------------------------------------------------
        # TEST PREDICTIONS
        # -----------------------------------------------------

        predictions = (
            fitted_model.predict(
                X_test
            )
        )

        # -----------------------------------------------------
        # ACCURACY
        # -----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        # -----------------------------------------------------
        # PRECISION
        # -----------------------------------------------------

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # -----------------------------------------------------
        # RECALL
        # -----------------------------------------------------

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # -----------------------------------------------------
        # F1 SCORE
        # -----------------------------------------------------

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # -----------------------------------------------------
        # ROC-AUC
        # -----------------------------------------------------

        roc_auc = get_roc_auc(
            fitted_model,
            X_test,
            y_test,
        )

        # -----------------------------------------------------
        # CROSS-VALIDATION SCORE
        # -----------------------------------------------------

        cv_accuracy = model.best_score_

        # -----------------------------------------------------
        # SAVE RESULTS
        # -----------------------------------------------------

        results.append(
            [
                name,
                accuracy,
                precision,
                recall,
                f1,
                roc_auc,
                cv_accuracy,
                model.best_params_.copy(),
            ]
        )

        # -----------------------------------------------------
        # SELECT BEST MODEL USING CV ONLY
        # -----------------------------------------------------

        if cv_accuracy > best_cv_score:

            best_cv_score = cv_accuracy

            best_model = fitted_model

            best_model_name = name

            best_predictions = predictions

            best_hyperparameters = (
                model.best_params_.copy()
            )

            best_roc_auc = roc_auc

    # =========================================================
    # RESULTS DATAFRAME
    # =========================================================

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "CV Accuracy",
            "Best Hyperparameters",
        ],
    )

    # CV accuracy determines the ranking
    # because CV is used to select the winner.
    results_df = (
        results_df
        .sort_values(
            "CV Accuracy",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # RETURN
    # =========================================================

    return {

        # Actual fitted winning estimator.
        "best_model": best_model,

        # Name of the winning model.
        "best_model_name": best_model_name,

        # Predictions made by the winning model.
        "best_predictions": best_predictions,

        # Hyperparameters selected by RandomizedSearchCV.
        "best_hyperparameters": (
            best_hyperparameters
        ),

        # ROC-AUC belonging to the winning model.
        "best_roc_auc": best_roc_auc,

        # Complete model comparison.
        "results": results_df,
    }