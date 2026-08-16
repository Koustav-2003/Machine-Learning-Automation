import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
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


def run_regression(train, test, target_column):

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

    CV = KFold(
        n_splits=5,
        shuffle=True,
        random_state=0,
    )

    # =========================================================
    # MODELS + HYPERPARAMETER SEARCH
    # =========================================================

    models = {

        "Simple Linear Regression": GridSearchCV(
            LinearRegression(),
            {
                "fit_intercept": [True, False],
            },
            cv=CV,
            scoring="r2",
            n_jobs=-1,
        ),

        "Multiple Linear Regression": GridSearchCV(
            LinearRegression(),
            {
                "fit_intercept": [True, False],
                "positive": [False, True],
            },
            cv=CV,
            scoring="r2",
            n_jobs=-1,
        ),

        "Polynomial Regression": GridSearchCV(
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
                "poly__degree": [2, 3],
                "linear__fit_intercept": [True, False],
            },
            cv=CV,
            scoring="r2",
            n_jobs=-1,
        ),

        "SVR": GridSearchCV(
            SVR(),
            {
                "kernel": ["poly", "linear"],
                "C": [1, 10, 100],
                "epsilon": [0.01, 0.1, 0.2],
                "gamma": ["scale", "auto"],
                "degree": [2, 3],
                "coef0": [0.0, 1.0],
            },
            cv=CV,
            scoring="r2",
            n_jobs=-1,
        ),

        "Decision Tree Regression": GridSearchCV(
            DecisionTreeRegressor(
                random_state=0
            ),
            {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            cv=CV,
            scoring="r2",
            n_jobs=-1,
        ),

        "Random Forest Regression": GridSearchCV(
            RandomForestRegressor(
                random_state=0
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            cv=CV,
            scoring="r2",
            n_jobs=-1,
        ),

        "XGBoost Regression": GridSearchCV(
            XGBRegressor(
                random_state=0,
                objective="reg:squarederror",
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
            cv=CV,
            scoring="r2",
            n_jobs=-1,
        ),

        "LightGBM Regression": GridSearchCV(
            LGBMRegressor(
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
            scoring="r2",
            n_jobs=-1,
        ),

        "CatBoost Regression": GridSearchCV(
            CatBoostRegressor(
                loss_function="RMSE",
                verbose=False,
                random_seed=0,
            ),
            {
                "iterations": [100, 200],
                "learning_rate": [0.05, 0.1],
                "depth": [4, 6, 8],
            },
            cv=CV,
            scoring="r2",
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

    # Best model is selected using CV R2.
    # Test-set performance is NEVER used to select the winner.
    best_cv_score = -np.inf

    for name, model in models.items():

        # -----------------------------------------------------
        # SIMPLE LINEAR REGRESSION
        # -----------------------------------------------------

        if name == "Simple Linear Regression":

            model.fit(
                X.iloc[:, [0]],
                y,
            )

            fitted_model = model.best_estimator_

            predictions = fitted_model.predict(
                X_test.iloc[:, [0]]
            )

        # -----------------------------------------------------
        # ALL OTHER MODELS
        # -----------------------------------------------------

        else:

            model.fit(
                X,
                y,
            )

            fitted_model = model.best_estimator_

            predictions = fitted_model.predict(
                X_test
            )

        # =====================================================
        # TEST METRICS
        # =====================================================

        test_r2 = r2_score(
            y_test,
            predictions,
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        # =====================================================
        # CV SCORE
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

            # This is the actual fitted winning model.
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

    # CV performance determines the ranking.
    results_df = results_df.sort_values(
        "CV R2",
        ascending=False,
    ).reset_index(drop=True)

    # =========================================================
    # RETURN
    # =========================================================

    return {
        # Actual fitted winning estimator.
        "best_model": best_model,

        # Name of the winning model.
        "best_model_name": best_model_name,

        # Predictions from the winning model.
        "best_predictions": best_predictions,

        # Hyperparameters selected by GridSearchCV.
        "best_hyperparameters": best_hyperparameters,

        # CV score used to select the winner.
        "best_cv_score": best_cv_score,

        # Complete model comparison.
        "results": results_df,
    }