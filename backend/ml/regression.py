import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def run_regression(
    train,
    test,
    target_column,
):
    """
    Train and evaluate regression models.

    Lite version:
        - 2-fold cross-validation
        - RandomizedSearchCV
        - 3 random parameter combinations
        - Reduced estimator ranges
        - Single CPU worker

    train.csv is used for:
        - Cross-validation
        - Hyperparameter tuning
        - Final model fitting

    test.csv is used only for:
        - Final test-set evaluation

    Returns:
        best_model
        best_model_name
        best_predictions
        best_hyperparameters
        best_cv_score
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
    # LITE CONFIGURATION
    # =========================================================

    # 2 folds instead of 5.
    CV = KFold(
        n_splits=2,
        shuffle=True,
        random_state=0,
    )

    # Only 3 random combinations are tested
    # for each hyperparameter search.
    RANDOM_SEARCH_ITERATIONS = 3

    # Single worker is safer for limited
    # hosted CPU environments.
    N_JOBS = 1

    # =========================================================
    # MODELS + HYPERPARAMETER SEARCH
    # =========================================================

    models = {

        # -----------------------------------------------------
        # Simple Linear Regression
        # -----------------------------------------------------

        "Simple Linear Regression": RandomizedSearchCV(
            LinearRegression(),
            {
                "fit_intercept": [
                    True,
                    False,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # Multiple Linear Regression
        # -----------------------------------------------------

        "Multiple Linear Regression": RandomizedSearchCV(
            LinearRegression(),
            {
                "fit_intercept": [
                    True,
                    False,
                ],
                "positive": [
                    False,
                    True,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # Polynomial Regression
        # -----------------------------------------------------

        "Polynomial Regression": RandomizedSearchCV(
            Pipeline(
                [
                    (
                        "poly",
                        PolynomialFeatures(
                            include_bias=False
                        ),
                    ),
                    (
                        "linear",
                        LinearRegression(),
                    ),
                ]
            ),
            {
                # Lite version keeps the same
                # compact degree search.
                "poly__degree": [
                    2,
                    3,
                ],
                "linear__fit_intercept": [
                    True,
                    False,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # Support Vector Regression
        # -----------------------------------------------------

        "SVR": RandomizedSearchCV(
            SVR(),
            {
                "kernel": [
                    "poly",
                    "linear",
                ],
                "C": [
                    1,
                    10,
                    100,
                ],
                "epsilon": [
                    0.01,
                    0.1,
                    0.2,
                ],
                "gamma": [
                    "scale",
                    "auto",
                ],
                "degree": [
                    2,
                    3,
                ],
                "coef0": [
                    0.0,
                    1.0,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # Decision Tree Regression
        # -----------------------------------------------------

        "Decision Tree Regression": RandomizedSearchCV(
            DecisionTreeRegressor(
                random_state=0
            ),
            {
                "max_depth": [
                    None,
                    5,
                    10,
                    20,
                ],
                "min_samples_split": [
                    2,
                    5,
                    10,
                ],
                "min_samples_leaf": [
                    1,
                    2,
                    4,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # Random Forest Regression
        # -----------------------------------------------------

        "Random Forest Regression": RandomizedSearchCV(
            RandomForestRegressor(
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
                    20,
                ],
                "min_samples_split": [
                    2,
                    5,
                ],
            },
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=CV,
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # XGBoost Regression
        # -----------------------------------------------------

        "XGBoost Regression": RandomizedSearchCV(
            XGBRegressor(
                random_state=0,
                objective="reg:squarederror",
                n_jobs=N_JOBS,
            ),
            {
                "n_estimators": [
                    50,
                    100,
                    150,
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
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # LightGBM Regression
        # -----------------------------------------------------

        "LightGBM Regression": RandomizedSearchCV(
            LGBMRegressor(
                random_state=0,
                verbosity=-1,
                n_jobs=N_JOBS,
            ),
            {
                "n_estimators": [
                    50,
                    100,
                    150,
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
            scoring="r2",
            random_state=0,
            n_jobs=N_JOBS,
        ),

        # -----------------------------------------------------
        # CatBoost Regression
        # -----------------------------------------------------

        "CatBoost Regression": RandomizedSearchCV(
            CatBoostRegressor(
                loss_function="RMSE",
                verbose=False,
                random_seed=0,
                thread_count=N_JOBS,
            ),
            {
                "iterations": [
                    50,
                    100,
                    150,
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
            scoring="r2",
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
    best_hyperparameters = None
    best_predictions = None

    # Best model is selected using CV R2.
    # Test performance is NOT used to select
    # the winning model.
    best_cv_score = -np.inf

    # =========================================================
    # TRAIN EVERY MODEL
    # =========================================================

    for name, model in models.items():

        # -----------------------------------------------------
        # SIMPLE LINEAR REGRESSION
        # -----------------------------------------------------

        if name == "Simple Linear Regression":

            # Simple Linear Regression uses
            # only the first feature.
            model.fit(
                X.iloc[:, [0]],
                y,
            )

            fitted_model = (
                model.best_estimator_
            )

            predictions = (
                fitted_model.predict(
                    X_test.iloc[:, [0]]
                )
            )

        # -----------------------------------------------------
        # ALL OTHER MODELS
        # -----------------------------------------------------

        else:

            model.fit(
                X,
                y,
            )

            fitted_model = (
                model.best_estimator_
            )

            predictions = (
                fitted_model.predict(
                    X_test
                )
            )

        # =====================================================
        # TEST R2
        # =====================================================

        test_r2 = r2_score(
            y_test,
            predictions,
        )

        # =====================================================
        # MAE
        # =====================================================

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        # =====================================================
        # RMSE
        # =====================================================

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        # =====================================================
        # CROSS-VALIDATION SCORE
        # =====================================================

        cv_r2 = model.best_score_

        # =====================================================
        # SAVE RESULTS
        # =====================================================

        results.append(
            [
                name,
                test_r2,
                mae,
                rmse,
                cv_r2,
                model.best_params_.copy(),
            ]
        )

        # =====================================================
        # SELECT BEST MODEL USING CV ONLY
        # =====================================================

        if cv_r2 > best_cv_score:

            best_cv_score = cv_r2

            # Actual fitted winning estimator.
            # routes.py will save this object.
            best_model = fitted_model

            best_model_name = name

            best_hyperparameters = (
                model.best_params_.copy()
            )

            best_predictions = predictions

    # =========================================================
    # RESULTS DATAFRAME
    # =========================================================

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "R2 Score",
            "MAE",
            "RMSE",
            "CV R2",
            "Best Hyperparameters",
        ],
    )

    # Rank models using CV R2.
    # Test-set performance is not used
    # to determine the winner.
    results_df = (
        results_df
        .sort_values(
            "CV R2",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # =========================================================
    # RETURN
    # =========================================================

    return {

        # Actual fitted winning estimator.
        "best_model": best_model,

        # Name of winning model.
        "best_model_name": best_model_name,

        # Predictions made by winning model.
        "best_predictions": best_predictions,

        # Selected hyperparameters.
        "best_hyperparameters": (
            best_hyperparameters
        ),

        # CV score used to select winner.
        "best_cv_score": best_cv_score,

        # Complete model comparison.
        "results": results_df,
    }