from io import BytesIO
import math
import multiprocessing
import queue
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import FileResponse

from sklearn.metrics import balanced_accuracy_score

from backend.ml import (
    run_classification,
    run_regression,
    run_clustering,
)

from backend.utils import (
    ValidationError,
    validate_supervised_data,
    validate_classification_targets,
    validate_regression_target,
    validate_unsupervised_data,
    detect_task,
    format_classification_metrics,
    format_regression_metrics,
    format_clustering_metrics,
)

from backend.utils.model_saver import (
    save_model,
    save_metadata,
)


router = APIRouter()


# =============================================================
# TRAINING JOB STORAGE
# =============================================================

# Stores running/background training processes.
#
# Example:
#
# {
#     "job_id": {
#         "process": <Process>,
#         "result_queue": <Queue>,
#         "status": "running",
#         "learning_type": "supervised"
#     }
# }
#
# The actual ML training happens in a separate process.

JOBS = {}


# =============================================================
# JSON SAFETY
# =============================================================

def make_json_safe(obj):
    """
    Recursively convert NumPy / pandas values and invalid
    floating-point values into JSON-safe Python values.

    NaN, +inf and -inf become None.
    """

    if isinstance(obj, dict):

        return {
            key: make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):

        return [
            make_json_safe(value)
            for value in obj
        ]

    if isinstance(obj, tuple):

        return [
            make_json_safe(value)
            for value in obj
        ]

    if isinstance(obj, np.integer):

        return int(obj)

    if isinstance(obj, np.floating):

        value = float(obj)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(obj, float):

        if not math.isfinite(obj):
            return None

        return obj

    if isinstance(obj, np.ndarray):

        return make_json_safe(
            obj.tolist()
        )

    return obj


# =============================================================
# CSV READER
# =============================================================

async def read_csv_file(
    file: UploadFile,
    name: str,
) -> pd.DataFrame:

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=f"{name} was not provided.",
        )

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail=f"{name} must be a CSV file.",
        )

    try:

        contents = await file.read()

        return pd.read_csv(
            BytesIO(contents)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read {name}: {str(exc)}"
            ),
        )


# =============================================================
# TRAINING WORKER
# =============================================================

def training_worker(
    train,
    test,
    learning_type,
    target_column,
    result_queue,
):
    """
    Executes the complete ML workflow in a separate process.

    The process can therefore be terminated by the
    /train/cancel/{job_id} endpoint.
    """

    try:

        learning_type = (
            learning_type
            .strip()
            .lower()
        )

        # =====================================================
        # SUPERVISED LEARNING
        # =====================================================

        if learning_type == "supervised":

            # -------------------------------------------------
            # COMMON VALIDATION
            # -------------------------------------------------

            validation_info = (
                validate_supervised_data(
                    train,
                    test,
                    target_column,
                )
            )

            # -------------------------------------------------
            # DETECT CLASSIFICATION / REGRESSION
            # -------------------------------------------------

            task = detect_task(
                train[target_column]
            )

            # =================================================
            # CLASSIFICATION
            # =================================================

            if task == "classification":

                validate_classification_targets(
                    train,
                    test,
                    target_column,
                )

                # ---------------------------------------------
                # RUN CLASSIFICATION
                # ---------------------------------------------

                output = run_classification(
                    train,
                    test,
                    target_column,
                )

                results = (
                    output["results"]
                    .copy()
                )

                # ---------------------------------------------
                # CONVERT METRICS TO PERCENTAGES
                # ---------------------------------------------

                percentage_columns = [
                    "Accuracy",
                    "Balanced Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Weighted Precision",
                    "Weighted Recall",
                    "Weighted F1",
                    "ROC-AUC",
                    "CV Accuracy",
                ]

                for column in percentage_columns:

                    if column in results.columns:

                        results[column] = (
                            results[column].apply(
                                lambda value: (
                                    None
                                    if pd.isna(value)
                                    else round(
                                        float(value) * 100,
                                        2,
                                    )
                                )
                            )
                        )

                # ---------------------------------------------
                # BEST MODEL
                # ---------------------------------------------

                best_model_name = (
                    output["best_model_name"]
                )

                best_rows = results[
                    results["Model"]
                    == best_model_name
                ]

                if best_rows.empty:

                    raise ValueError(
                        "Best classification model "
                        "was not found in results."
                    )

                best_row = (
                    best_rows.iloc[0]
                )

                # ---------------------------------------------
                # ACCURACY
                # ---------------------------------------------

                accuracy = None

                if "Accuracy" in best_row.index:

                    if not pd.isna(
                        best_row["Accuracy"]
                    ):

                        accuracy = (
                            float(
                                best_row["Accuracy"]
                            ) / 100
                        )

                # ---------------------------------------------
                # BALANCED ACCURACY
                # ---------------------------------------------

                balanced_accuracy = None

                if (
                    "best_predictions"
                    in output
                    and output["best_predictions"]
                    is not None
                ):

                    try:

                        predictions = (
                            output[
                                "best_predictions"
                            ]
                        )

                        y_test = (
                            test[target_column]
                        )

                        balanced_accuracy = (
                            balanced_accuracy_score(
                                y_test,
                                predictions,
                            )
                        )

                    except Exception:

                        balanced_accuracy = None

                elif (
                    "Balanced Accuracy"
                    in best_row.index
                    and not pd.isna(
                        best_row[
                            "Balanced Accuracy"
                        ]
                    )
                ):

                    balanced_accuracy = (
                        float(
                            best_row[
                                "Balanced Accuracy"
                            ]
                        ) / 100
                    )

                # ---------------------------------------------
                # PRECISION
                # ---------------------------------------------

                if "Precision" in best_row.index:

                    precision = (
                        None
                        if pd.isna(
                            best_row["Precision"]
                        )
                        else float(
                            best_row["Precision"]
                        ) / 100
                    )

                elif (
                    "Weighted Precision"
                    in best_row.index
                ):

                    precision = (
                        None
                        if pd.isna(
                            best_row[
                                "Weighted Precision"
                            ]
                        )
                        else float(
                            best_row[
                                "Weighted Precision"
                            ]
                        ) / 100
                    )

                else:

                    precision = None

                # ---------------------------------------------
                # RECALL
                # ---------------------------------------------

                if "Recall" in best_row.index:

                    recall = (
                        None
                        if pd.isna(
                            best_row["Recall"]
                        )
                        else float(
                            best_row["Recall"]
                        ) / 100
                    )

                elif (
                    "Weighted Recall"
                    in best_row.index
                ):

                    recall = (
                        None
                        if pd.isna(
                            best_row[
                                "Weighted Recall"
                            ]
                        )
                        else float(
                            best_row[
                                "Weighted Recall"
                            ]
                        ) / 100
                    )

                else:

                    recall = None

                # ---------------------------------------------
                # F1
                # ---------------------------------------------

                if "F1 Score" in best_row.index:

                    f1 = (
                        None
                        if pd.isna(
                            best_row["F1 Score"]
                        )
                        else float(
                            best_row["F1 Score"]
                        ) / 100
                    )

                elif (
                    "Weighted F1"
                    in best_row.index
                ):

                    f1 = (
                        None
                        if pd.isna(
                            best_row[
                                "Weighted F1"
                            ]
                        )
                        else float(
                            best_row[
                                "Weighted F1"
                            ]
                        ) / 100
                    )

                else:

                    f1 = None

                # ---------------------------------------------
                # ROC-AUC
                # ---------------------------------------------

                roc_auc = None

                if (
                    "ROC-AUC"
                    in best_row.index
                    and not pd.isna(
                        best_row["ROC-AUC"]
                    )
                ):

                    roc_auc = (
                        float(
                            best_row["ROC-AUC"]
                        ) / 100
                    )

                # ---------------------------------------------
                # SAVE MODEL
                # ---------------------------------------------

                model_path = save_model(
                    output["best_model"],
                    "classification",
                )

                # ---------------------------------------------
                # FORMAT SCORES
                # ---------------------------------------------

                formatted_scores = (
                    format_classification_metrics(
                        accuracy=accuracy,
                        balanced_accuracy=(
                            balanced_accuracy
                        ),
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        roc_auc=roc_auc,
                    )
                )

                # ---------------------------------------------
                # ALL MODELS
                # ---------------------------------------------

                all_models = make_json_safe(
                    results.to_dict(
                        orient="records"
                    )
                )

                # ---------------------------------------------
                # METADATA
                # ---------------------------------------------

                metadata = {

                    "task": "classification",

                    "learning_type": "supervised",

                    "target_column": (
                        target_column
                    ),

                    "best_model": (
                        best_model_name
                    ),

                    "best_hyperparameters": (
                        output[
                            "best_hyperparameters"
                        ]
                    ),

                    "best_model_scores": (
                        formatted_scores
                    ),

                    "feature_columns": (
                        train.drop(
                            columns=[
                                target_column
                            ]
                        ).columns.tolist()
                    ),

                    "validation": (
                        validation_info
                    ),

                    "all_models": (
                        all_models
                    ),
                }

                metadata = make_json_safe(
                    metadata
                )

                metadata_path = (
                    save_metadata(
                        "classification",
                        metadata,
                    )
                )

                response = {

                    "status": "success",

                    "learning_type": "supervised",

                    "task": "classification",

                    "target_column": (
                        target_column
                    ),

                    "best_model": (
                        best_model_name
                    ),

                    "best_hyperparameters": (
                        output[
                            "best_hyperparameters"
                        ]
                    ),

                    "best_model_scores": (
                        formatted_scores
                    ),

                    "all_models": (
                        all_models
                    ),

                    "validation": (
                        validation_info
                    ),

                    "model_saved": True,

                    "model_path": (
                        model_path
                    ),

                    "metadata_saved": True,

                    "metadata_path": (
                        metadata_path
                    ),
                }

                result_queue.put(
                    make_json_safe(
                        response
                    )
                )

                return

            # =================================================
            # REGRESSION
            # =================================================

            validate_regression_target(
                train,
                target_column,
            )

            # ---------------------------------------------
            # RUN REGRESSION
            # ---------------------------------------------

            output = run_regression(
                train,
                test,
                target_column,
            )

            results = (
                output["results"]
                .copy()
            )

            # ---------------------------------------------
            # R2 -> PERCENTAGE
            # ---------------------------------------------

            if "R2 Score" in results.columns:

                results[
                    "R2 Score"
                ] = (
                    results[
                        "R2 Score"
                    ].apply(
                        lambda value: (
                            None
                            if pd.isna(value)
                            else round(
                                float(value) * 100,
                                2,
                            )
                        )
                    )
                )

            # ---------------------------------------------
            # CV R2 -> PERCENTAGE
            # ---------------------------------------------

            if "CV R2" in results.columns:

                results[
                    "CV R2"
                ] = (
                    results[
                        "CV R2"
                    ].apply(
                        lambda value: (
                            None
                            if pd.isna(value)
                            else round(
                                float(value) * 100,
                                2,
                            )
                        )
                    )
                )

            # ---------------------------------------------
            # BEST MODEL
            # ---------------------------------------------

            best_model_name = (
                output["best_model_name"]
            )

            best_rows = results[
                results["Model"]
                == best_model_name
            ]

            if best_rows.empty:

                raise ValueError(
                    "Best regression model "
                    "was not found in results."
                )

            best_row = (
                best_rows.iloc[0]
            )

            # ---------------------------------------------
            # SAVE MODEL
            # ---------------------------------------------

            model_path = save_model(
                output["best_model"],
                "regression",
            )

            # ---------------------------------------------
            # FORMAT SCORES
            # ---------------------------------------------

            formatted_scores = (
                format_regression_metrics(
                    r2=(
                        float(
                            best_row[
                                "R2 Score"
                            ]
                        ) / 100
                    ),

                    mae=(
                        float(
                            best_row["MAE"]
                        )
                    ),

                    rmse=(
                        float(
                            best_row["RMSE"]
                        )
                    ),
                )
            )

            # ---------------------------------------------
            # ALL MODELS
            # ---------------------------------------------

            all_models = make_json_safe(
                results.to_dict(
                    orient="records"
                )
            )

            # ---------------------------------------------
            # METADATA
            # ---------------------------------------------

            metadata = {

                "task": "regression",

                "learning_type": "supervised",

                "target_column": (
                    target_column
                ),

                "best_model": (
                    best_model_name
                ),

                "best_hyperparameters": (
                    output[
                        "best_hyperparameters"
                    ]
                ),

                "best_model_scores": (
                    formatted_scores
                ),

                "feature_columns": (
                    train.drop(
                        columns=[
                            target_column
                        ]
                    ).columns.tolist()
                ),

                "validation": (
                    validation_info
                ),

                "all_models": (
                    all_models
                ),
            }

            metadata = make_json_safe(
                metadata
            )

            metadata_path = (
                save_metadata(
                    "regression",
                    metadata,
                )
            )

            response = {

                "status": "success",

                "learning_type": "supervised",

                "task": "regression",

                "target_column": (
                    target_column
                ),

                "best_model": (
                    best_model_name
                ),

                "best_hyperparameters": (
                    output[
                        "best_hyperparameters"
                    ]
                ),

                "best_model_scores": (
                    formatted_scores
                ),

                "all_models": (
                    all_models
                ),

                "validation": (
                    validation_info
                ),

                "model_saved": True,

                "model_path": (
                    model_path
                ),

                "metadata_saved": True,

                "metadata_path": (
                    metadata_path
                ),
            }

            result_queue.put(
                make_json_safe(
                    response
                )
            )

            return

        # =====================================================
        # UNSUPERVISED / CLUSTERING
        # =====================================================

        validation_info = (
            validate_unsupervised_data(
                train,
                test,
            )
        )

        # ---------------------------------------------
        # RUN CLUSTERING
        # ---------------------------------------------

        output = run_clustering(
            train,
            test,
        )

        results = (
            output["results"]
            .copy()
        )

        # ---------------------------------------------
        # TRAIN SILHOUETTE -> %
        # ---------------------------------------------

        if (
            "Train Silhouette Score"
            in results.columns
        ):

            results[
                "Train Silhouette Score"
            ] = (
                results[
                    "Train Silhouette Score"
                ].apply(
                    lambda value: (
                        None
                        if pd.isna(value)
                        else round(
                            float(value) * 100,
                            2,
                        )
                    )
                )
            )

        # ---------------------------------------------
        # TEST SILHOUETTE -> %
        # ---------------------------------------------

        if (
            "Test Silhouette Score"
            in results.columns
        ):

            results[
                "Test Silhouette Score"
            ] = (
                results[
                    "Test Silhouette Score"
                ].apply(
                    lambda value: (
                        None
                        if pd.isna(value)
                        else round(
                            float(value) * 100,
                            2,
                        )
                    )
                )
            )

        # ---------------------------------------------
        # BEST MODEL
        # ---------------------------------------------

        best_model_name = (
            output["best_model_name"]
        )

        best_rows = results[
            results["Model"]
            == best_model_name
        ]

        if best_rows.empty:

            raise ValueError(
                "Best clustering model "
                "was not found in results."
            )

        best_row = (
            best_rows.iloc[0]
        )

        # ---------------------------------------------
        # BEST K
        # ---------------------------------------------

        best_k = output.get(
            "best_k"
        )

        if best_k is None:

            best_k = best_row.get(
                "Best Number of Clusters"
            )

        # ---------------------------------------------
        # TEST SILHOUETTE
        # ---------------------------------------------

        test_silhouette = (
            best_row.get(
                "Test Silhouette Score"
            )
        )

        if (
            test_silhouette is None
            or pd.isna(
                test_silhouette
            )
        ):

            formatted_silhouette = None

        else:

            formatted_silhouette = (
                float(
                    test_silhouette
                ) / 100
            )

        # ---------------------------------------------
        # SAVE MODEL
        # ---------------------------------------------

        model_path = save_model(
            output["best_model"],
            "clustering",
        )

        # ---------------------------------------------
        # FORMAT SCORE
        # ---------------------------------------------

        formatted_scores = (
            format_clustering_metrics(
                silhouette=(
                    formatted_silhouette
                ),
            )
        )

        # ---------------------------------------------
        # ALL MODELS
        # ---------------------------------------------

        all_models = make_json_safe(
            results.to_dict(
                orient="records"
            )
        )

        # ---------------------------------------------
        # METADATA
        # ---------------------------------------------

        metadata = {

            "task": "clustering",

            "learning_type": "unsupervised",

            "target_column": None,

            "best_model": (
                best_model_name
            ),

            "best_k": (
                best_k
            ),

            "best_model_scores": (
                formatted_scores
            ),

            "feature_columns": (
                train.columns.tolist()
            ),

            "validation": (
                validation_info
            ),

            "all_models": (
                all_models
            ),
        }

        metadata = make_json_safe(
            metadata
        )

        metadata_path = (
            save_metadata(
                "clustering",
                metadata,
            )
        )

        response = {

            "status": "success",

            "learning_type": "unsupervised",

            "task": "clustering",

            "target_column": None,

            "best_model": (
                best_model_name
            ),

            "best_k": (
                best_k
            ),

            "best_model_scores": (
                formatted_scores
            ),

            "all_models": (
                all_models
            ),

            "validation": (
                validation_info
            ),

            "model_saved": True,

            "model_path": (
                model_path
            ),

            "metadata_saved": True,

            "metadata_path": (
                metadata_path
            ),
        }

        result_queue.put(
            make_json_safe(
                response
            )
        )

    # =========================================================
    # VALIDATION ERROR
    # =========================================================

    except ValidationError as exc:

        result_queue.put(
            {
                "status": "error",
                "error_type": "validation",
                "detail": str(exc),
            }
        )

    # =========================================================
    # VALUE ERROR
    # =========================================================

    except ValueError as exc:

        result_queue.put(
            {
                "status": "error",
                "error_type": "value",
                "detail": str(exc),
            }
        )

    # =========================================================
    # GENERAL ERROR
    # =========================================================

    except Exception as exc:

        result_queue.put(
            {
                "status": "error",
                "error_type": "training",
                "detail": (
                    f"Model training failed: {str(exc)}"
                ),
            }
        )


# =============================================================
# START TRAINING
# =============================================================

@router.post("/train/start")
async def start_training(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    learning_type: str = Form(...),
    target_column: str | None = Form(None),
):
    """
    Start ML training in a separate process.

    Returns immediately with a job_id.
    """

    learning_type = (
        learning_type
        .strip()
        .lower()
    )

    if learning_type not in {
        "supervised",
        "unsupervised",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "learning_type must be "
                "'supervised' or 'unsupervised'."
            ),
        )

    # ---------------------------------------------------------
    # READ FILES BEFORE STARTING PROCESS
    # ---------------------------------------------------------

    train = await read_csv_file(
        train_file,
        "train.csv",
    )

    test = await read_csv_file(
        test_file,
        "test.csv",
    )

    # ---------------------------------------------------------
    # TARGET REQUIREMENT
    # ---------------------------------------------------------

    if learning_type == "supervised":

        if (
            not target_column
            or not target_column.strip()
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "target_column is required for "
                    "supervised learning."
                ),
            )

        target_column = (
            target_column.strip()
        )

    else:

        target_column = None

    # ---------------------------------------------------------
    # PRE-VALIDATE DATA
    # ---------------------------------------------------------

    try:

        if learning_type == "supervised":

            validate_supervised_data(
                train,
                test,
                target_column,
            )

            task = detect_task(
                train[target_column]
            )

            if task == "classification":

                validate_classification_targets(
                    train,
                    test,
                    target_column,
                )

            else:

                validate_regression_target(
                    train,
                    target_column,
                )

        else:

            validate_unsupervised_data(
                train,
                test,
            )

    except ValidationError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # ---------------------------------------------------------
    # CREATE JOB
    # ---------------------------------------------------------

    job_id = str(
        uuid.uuid4()
    )

    result_queue = (
        multiprocessing.Queue()
    )

    # ---------------------------------------------------------
    # CREATE PROCESS
    # ---------------------------------------------------------

    process = multiprocessing.Process(
        target=training_worker,
        args=(
            train,
            test,
            learning_type,
            target_column,
            result_queue,
        ),
    )

    # ---------------------------------------------------------
    # STORE JOB
    # ---------------------------------------------------------

    JOBS[job_id] = {

        "process": process,

        "result_queue": result_queue,

        "status": "running",

        "learning_type": learning_type,
    }

    # ---------------------------------------------------------
    # START
    # ---------------------------------------------------------

    process.start()

    return {

        "status": "started",

        "job_id": job_id,

        "message": (
            "Model training started."
        ),
    }


# =============================================================
# TRAINING STATUS
# =============================================================

@router.get("/train/status/{job_id}")
async def training_status(
    job_id: str,
):
    """
    Get the current status of a training job.
    """

    if job_id not in JOBS:

        raise HTTPException(
            status_code=404,
            detail="Training job not found.",
        )

    job = JOBS[job_id]

    process = job["process"]

    result_queue = (
        job["result_queue"]
    )

    # ---------------------------------------------------------
    # CHECK FOR RESULT
    # ---------------------------------------------------------

    try:

        result = (
            result_queue.get_nowait()
        )

        # ---------------------------------------------
        # TRAINING ERROR
        # ---------------------------------------------

        if result.get(
            "status"
        ) == "error":

            job["status"] = "failed"

            return {

                "status": "failed",

                "job_id": job_id,

                "error_type": result.get(
                    "error_type"
                ),

                "detail": result.get(
                    "detail"
                ),
            }

        # ---------------------------------------------
        # SUCCESS
        # ---------------------------------------------

        job["status"] = "completed"

        return {

            "status": "completed",

            "job_id": job_id,

            "result": make_json_safe(
                result
            ),
        }

    except queue.Empty:

        pass

    # ---------------------------------------------------------
    # PROCESS STILL RUNNING
    # ---------------------------------------------------------

    if process.is_alive():

        return {

            "status": "running",

            "job_id": job_id,
        }

    # ---------------------------------------------------------
    # PROCESS FINISHED WITHOUT RESULT
    # ---------------------------------------------------------

    if job["status"] == "cancelled":

        return {

            "status": "cancelled",

            "job_id": job_id,
        }

    job["status"] = "failed"

    return {

        "status": "failed",

        "job_id": job_id,

        "detail": (
            "Training process ended unexpectedly."
        ),
    }


# =============================================================
# CANCEL TRAINING
# =============================================================

@router.post("/train/cancel/{job_id}")
async def cancel_training(
    job_id: str,
):
    """
    Immediately terminate a running training process.
    """

    if job_id not in JOBS:

        raise HTTPException(
            status_code=404,
            detail="Training job not found.",
        )

    job = JOBS[job_id]

    process = job["process"]

    # ---------------------------------------------------------
    # ALREADY FINISHED
    # ---------------------------------------------------------

    if not process.is_alive():

        if job["status"] == "completed":

            return {

                "status": "completed",

                "job_id": job_id,

                "message": (
                    "Training had already completed."
                ),
            }

        return {

            "status": job["status"],

            "job_id": job_id,
        }

    # ---------------------------------------------------------
    # TERMINATE PROCESS
    # ---------------------------------------------------------

    process.terminate()

    process.join(
        timeout=5
    )

    # ---------------------------------------------------------
    # FORCE KILL IF NECESSARY
    # ---------------------------------------------------------

    if process.is_alive():

        process.kill()

        process.join(
            timeout=5
        )

    job["status"] = "cancelled"

    return {

        "status": "cancelled",

        "job_id": job_id,

        "message": (
            "Training was cancelled."
        ),
    }


# =============================================================
# DOWNLOAD TRAINED MODEL
# =============================================================

@router.get("/download-model/{task}")
async def download_model(
    task: str,
):
    """
    Download the trained .pkl model.

    Supported tasks:
        classification
        regression
        clustering
    """

    task = task.strip().lower()

    model_files = {

        "classification":
            "classification_model.pkl",

        "regression":
            "regression_model.pkl",

        "clustering":
            "clustering_model.pkl",
    }

    if task not in model_files:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid task. Supported tasks are "
                "classification, regression and clustering."
            ),
        )

    # ---------------------------------------------------------
    # SAVED MODELS DIRECTORY
    # ---------------------------------------------------------

    model_path = (
        Path(__file__).resolve().parents[2]
        / "saved_models"
        / model_files[task]
    )

    # ---------------------------------------------------------
    # CHECK FILE EXISTS
    # ---------------------------------------------------------

    if not model_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"No saved {task} model was found. "
                "Run model training first."
            ),
        )

    # ---------------------------------------------------------
    # RETURN FILE
    # ---------------------------------------------------------

    return FileResponse(
        path=str(model_path),
        filename=model_files[task],
        media_type="application/octet-stream",
    )


# =============================================================
# LEGACY / TRAIN ENDPOINT
# =============================================================

@router.post("/train")
async def train_models_legacy(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    learning_type: str = Form(...),
    target_column: str | None = Form(None),
):
    """
    Legacy synchronous endpoint.

    Kept for compatibility.

    The Streamlit frontend should use:

        POST /train/start
        GET  /train/status/{job_id}
        POST /train/cancel/{job_id}

    This endpoint does not support interactive cancellation.
    """

    # ---------------------------------------------------------
    # READ FILES
    # ---------------------------------------------------------

    train = await read_csv_file(
        train_file,
        "train.csv",
    )

    test = await read_csv_file(
        test_file,
        "test.csv",
    )

    learning_type = (
        learning_type
        .strip()
        .lower()
    )

    if learning_type not in {
        "supervised",
        "unsupervised",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "learning_type must be "
                "'supervised' or 'unsupervised'."
            ),
        )

    if learning_type == "supervised":

        if (
            not target_column
            or not target_column.strip()
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "target_column is required for "
                    "supervised learning."
                ),
            )

        target_column = (
            target_column.strip()
        )

    else:

        target_column = None

    # ---------------------------------------------------------
    # LOCAL QUEUE
    # ---------------------------------------------------------

    result_queue = (
        multiprocessing.Queue()
    )

    # ---------------------------------------------------------
    # RUN WORKER DIRECTLY
    # ---------------------------------------------------------

    training_worker(
        train,
        test,
        learning_type,
        target_column,
        result_queue,
    )

    try:

        result = (
            result_queue.get_nowait()
        )

    except queue.Empty:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model training failed without "
                "returning a result."
            ),
        )

    # ---------------------------------------------------------
    # ERROR
    # ---------------------------------------------------------

    if result.get(
        "status"
    ) == "error":

        error_type = result.get(
            "error_type"
        )

        if error_type == "validation":

            raise HTTPException(
                status_code=400,
                detail=result.get(
                    "detail"
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=result.get(
                "detail"
            ),
        )

    return make_json_safe(
        result
    )