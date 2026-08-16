import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV

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


def run_classification(train, test, target_column):

    # =========================================================
    # FEATURES / TARGET
    # =========================================================

    X = train.drop(columns=[target_column])
    y = train[target_column]

    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    # =========================================================
    # CROSS VALIDATION
    # =========================================================

    CV = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=0,
    )

    # =========================================================
    # MODELS + HYPERPARAMETER SEARCH
    # =========================================================

    models = {

        "Logistic Regression": GridSearchCV(
            LogisticRegression(
                max_iter=2000
            ),
            {
                "C": [0.1, 1, 10, 100],
                "solver": ["lbfgs"],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),

        "KNN": GridSearchCV(
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),

        "SVM": GridSearchCV(
            SVC(),
            {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),

        "Decision Tree": GridSearchCV(
            DecisionTreeClassifier(
                random_state=0
            ),
            {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),

        "Random Forest": GridSearchCV(
            RandomForestClassifier(
                random_state=0
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),

        "XGBoost": GridSearchCV(
            XGBClassifier(
                random_state=0,
                eval_metric="logloss",
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),

        "LightGBM": GridSearchCV(
            LGBMClassifier(
                random_state=0,
                verbosity=-1,
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 10],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),

        "CatBoost": GridSearchCV(
            CatBoostClassifier(
                verbose=False,
                random_seed=0,
            ),
            {
                "iterations": [100, 200],
                "learning_rate": [0.05, 0.1],
                "depth": [4, 6, 8],
            },
            cv=CV,
            scoring="accuracy",
            n_jobs=-1,
        ),
    }

    # =========================================================
    # TRAIN + EVALUATE
    # =========================================================

    results = []

    best_model = None
    best_model_name = None
    best_hyperparameters = None
    best_predictions = None
    best_roc_auc = None

    # IMPORTANT:
    # Select the best model using CV accuracy.
    # The test set is NEVER used to select the winner.
    best_cv_score = -np.inf

    for name, model in models.items():

        # =====================================================
        # TRAIN
        # =====================================================

        model.fit(
            X,
            y,
        )

        fitted_model = model.best_estimator_

        # =====================================================
        # TEST PREDICTIONS
        # =====================================================

        predictions = fitted_model.predict(
            X_test
        )

        # =====================================================
        # TEST METRICS
        # =====================================================

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # =====================================================
        # ROC-AUC
        # =====================================================

        roc_auc = None

        try:

            number_of_classes = len(
                np.unique(y_test)
            )

            # -------------------------------------------------
            # MODELS WITH predict_proba()
            # -------------------------------------------------

            if hasattr(
                fitted_model,
                "predict_proba",
            ):

                probabilities = (
                    fitted_model.predict_proba(
                        X_test
                    )
                )

                if number_of_classes == 2:

                    roc_auc = roc_auc_score(
                        y_test,
                        probabilities[:, 1],
                    )

                else:

                    roc_auc = roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                    )

            # -------------------------------------------------
            # MODELS WITHOUT predict_proba()
            # -------------------------------------------------

            elif hasattr(
                fitted_model,
                "decision_function",
            ):

                decision_scores = (
                    fitted_model.decision_function(
                        X_test
                    )
                )

                if number_of_classes == 2:

                    roc_auc = roc_auc_score(
                        y_test,
                        decision_scores,
                    )

                else:

                    roc_auc = roc_auc_score(
                        y_test,
                        decision_scores,
                        multi_class="ovr",
                        average="weighted",
                    )

        except Exception:

            # ROC-AUC is allowed to be unavailable
            # for models/datasets where it cannot be
            # calculated safely.
            roc_auc = None

        # =====================================================
        # CV SCORE
        # =====================================================

        cv_accuracy = model.best_score_

        # =====================================================
        # SAVE RESULTS
        # =====================================================

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

        # =====================================================
        # SELECT BEST MODEL USING CV ONLY
        # =====================================================

        if cv_accuracy > best_cv_score:

            best_cv_score = cv_accuracy

            best_model = fitted_model

            best_model_name = name

            best_hyperparameters = (
                model.best_params_.copy()
            )

            best_predictions = predictions

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

    # Sort by CV accuracy because CV determines
    # the winning model.
    results_df = results_df.sort_values(
        "CV Accuracy",
        ascending=False,
    ).reset_index(drop=True)

    # =========================================================
    # RETURN
    # =========================================================

    return {

        "best_model": best_model,

        "best_model_name": best_model_name,

        "best_predictions": best_predictions,

        "best_hyperparameters": (
            best_hyperparameters
        ),

        "best_roc_auc": best_roc_auc,

        "results": results_df,
    }