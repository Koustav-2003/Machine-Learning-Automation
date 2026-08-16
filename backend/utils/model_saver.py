import json
import os
import joblib


# =============================================================
# SAVED MODELS DIRECTORY
# =============================================================

MODELS_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    ),
    "saved_models"
)


# =============================================================
# SAVE MODEL
# =============================================================

def save_model(model, task):
    """
    Save the trained best model to disk.

    Args:
        model:
            Trained sklearn-compatible model.

        task:
            Type of ML task:
            regression
            classification
            clustering

    Returns:
        Path of the saved model.
    """

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    task = task.lower().strip()

    filename = f"{task}_model.pkl"

    model_path = os.path.join(
        MODELS_DIR,
        filename
    )

    joblib.dump(
        model,
        model_path
    )

    return model_path


# =============================================================
# LOAD MODEL
# =============================================================

def load_model(task):
    """
    Load a previously saved model.

    Args:
        task:
            regression
            classification
            clustering

    Returns:
        Loaded model.
    """

    task = task.lower().strip()

    filename = f"{task}_model.pkl"

    model_path = os.path.join(
        MODELS_DIR,
        filename
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Saved {task} model not found."
        )

    return joblib.load(
        model_path
    )


# =============================================================
# SAVE METADATA
# =============================================================

def save_metadata(task, metadata):
    """
    Save metadata associated with a trained model.

    Args:
        task:
            regression
            classification
            clustering

        metadata:
            Dictionary containing model information.

    Returns:
        Path of the metadata file.
    """

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    task = task.lower().strip()

    filename = f"{task}_metadata.json"

    metadata_path = os.path.join(
        MODELS_DIR,
        filename
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    return metadata_path


# =============================================================
# LOAD METADATA
# =============================================================

def load_metadata(task):
    """
    Load metadata associated with a trained model.

    Args:
        task:
            regression
            classification
            clustering

    Returns:
        Metadata dictionary.
    """

    task = task.lower().strip()

    filename = f"{task}_metadata.json"

    metadata_path = os.path.join(
        MODELS_DIR,
        filename
    )

    if not os.path.exists(metadata_path):

        raise FileNotFoundError(
            f"Metadata for {task} model not found."
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)