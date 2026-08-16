import hashlib
import time

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_BASE_URL = "http://127.0.0.1:8000"

START_URL = f"{BACKEND_BASE_URL}/train/start"
STATUS_URL = f"{BACKEND_BASE_URL}/train/status"
CANCEL_URL = f"{BACKEND_BASE_URL}/train/cancel"
DOWNLOAD_MODEL_URL = f"{BACKEND_BASE_URL}/download-model"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Machine Learning Automation",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "learning_type": "Supervised",
    "job_id": None,
    "training_running": False,
    "result": None,
    "analysis_complete": False,
    "input_signature": None,
    "last_learning_type": "Supervised",
    "cancel_message": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clear_analysis():

    st.session_state.job_id = None
    st.session_state.training_running = False
    st.session_state.result = None
    st.session_state.analysis_complete = False
    st.session_state.input_signature = None
    st.session_state.cancel_message = None


def file_signature(uploaded_file):

    if uploaded_file is None:

        return None

    try:

        contents = uploaded_file.getvalue()

        return (
            uploaded_file.name,
            len(contents),
            hashlib.md5(
                contents
            ).hexdigest(),
        )

    except Exception:

        return uploaded_file.name


def build_input_signature(
    learning_type,
    train_file,
    test_file,
    target_column,
):

    return (
        learning_type,
        file_signature(train_file),
        file_signature(test_file),
        target_column,
    )


def backend_error(response):

    try:

        return response.json().get(
            "detail",
            "Unknown backend error.",
        )

    except Exception:

        return (
            response.text
            or "Unknown backend error."
        )


def start_training(
    train_file,
    test_file,
    learning_type,
    target_column,
):

    files = {

        "train_file": (
            train_file.name,
            train_file.getvalue(),
            "text/csv",
        ),

        "test_file": (
            test_file.name,
            test_file.getvalue(),
            "text/csv",
        ),
    }

    data = {
        "learning_type": (
            learning_type.lower()
        ),
    }

    if learning_type == "Supervised":

        data["target_column"] = (
            target_column
        )

    response = requests.post(
        START_URL,
        files=files,
        data=data,
        timeout=120,
    )

    if response.status_code != 200:

        raise RuntimeError(
            backend_error(response)
        )

    payload = response.json()

    job_id = payload.get(
        "job_id"
    )

    if not job_id:

        raise RuntimeError(
            "Backend started the request "
            "but did not return a job_id."
        )

    return job_id


def get_training_status(job_id):

    response = requests.get(
        f"{STATUS_URL}/{job_id}",
        timeout=30,
    )

    if response.status_code != 200:

        raise RuntimeError(
            backend_error(response)
        )

    return response.json()


def cancel_training(job_id):

    response = requests.post(
        f"{CANCEL_URL}/{job_id}",
        timeout=30,
    )

    if response.status_code != 200:

        raise RuntimeError(
            backend_error(response)
        )

    return response.json()


def get_model_download(task):

    """
    Get the saved .pkl model from FastAPI.
    """

    response = requests.get(
        f"{DOWNLOAD_MODEL_URL}/{task}",
        timeout=60,
    )

    if response.status_code != 200:

        raise RuntimeError(
            backend_error(response)
        )

    return response.content


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 Machine Learning Automation"
)

st.write(
    "Automatically compare multiple machine learning "
    "models and find the best choice for your dataset."
)


# ============================================================
# LEARNING TYPE
# ============================================================

st.subheader(
    "1. Select Learning Type"
)

learning_type = st.radio(
    "What type of learning are you using?",
    [
        "Supervised",
        "Unsupervised",
    ],
    horizontal=True,
)


# ============================================================
# RESET WHEN LEARNING TYPE CHANGES
# ============================================================

if (
    learning_type
    != st.session_state.last_learning_type
):

    # Do not allow an old result from another learning
    # type to remain on screen.

    if st.session_state.training_running:

        st.warning(
            "A training operation is currently running. "
            "Please cancel it before switching learning type."
        )

    else:

        st.session_state.result = None
        st.session_state.analysis_complete = False
        st.session_state.job_id = None
        st.session_state.cancel_message = None
        st.session_state.input_signature = None

    st.session_state.last_learning_type = (
        learning_type
    )


# ============================================================
# PREPROCESSING WARNING
# ============================================================

st.warning(
    "⚠️ Please ensure your data is already pre-processed "
    "before uploading. This tool does not perform data "
    "preprocessing."
)


# ============================================================
# VARIABLES
# ============================================================

train_file = None
test_file = None
target_column = None
train_preview = None


# ============================================================
# SUPERVISED LEARNING
# ============================================================

if learning_type == "Supervised":

    st.subheader(
        "2. Upload Datasets"
    )

    train_file = st.file_uploader(
        "Training Dataset",
        type=["csv"],
        help=(
            "Upload your pre-processed "
            "training CSV file."
        ),
        key="supervised_train",
    )

    test_file = st.file_uploader(
        "Testing Dataset",
        type=["csv"],
        help=(
            "Upload your pre-processed "
            "testing CSV file."
        ),
        key="supervised_test",
    )

    # --------------------------------------------------------
    # TRAINING DATA PREVIEW
    # --------------------------------------------------------

    if train_file is not None:

        try:

            train_preview = pd.read_csv(
                train_file
            )

            st.write(
                f"Training dataset: "
                f"{train_preview.shape[0]} rows × "
                f"{train_preview.shape[1]} columns"
            )

            if len(
                train_preview.columns
            ) == 0:

                st.error(
                    "The training dataset "
                    "contains no columns."
                )

                train_preview = None

            else:

                # ------------------------------------------------
                # TARGET COLUMN
                #
                # Last column selected by default.
                # ------------------------------------------------

                target_column = st.selectbox(
                    "Target Column",
                    train_preview.columns,
                    index=(
                        len(
                            train_preview.columns
                        ) - 1
                    ),
                    help=(
                        "The last column is selected "
                        "by default. Change it if your "
                        "target column is different."
                    ),
                    key="supervised_target",
                )

        except Exception as error:

            st.error(
                f"Could not read the training CSV: {error}"
            )

            train_preview = None


# ============================================================
# UNSUPERVISED LEARNING
# ============================================================

else:

    st.subheader(
        "2. Upload Datasets"
    )

    train_file = st.file_uploader(
        "Training Dataset",
        type=["csv"],
        help=(
            "Upload your pre-processed "
            "training CSV file."
        ),
        key="unsupervised_train",
    )

    test_file = st.file_uploader(
        "Testing Dataset",
        type=["csv"],
        help=(
            "Upload your pre-processed "
            "testing CSV file."
        ),
        key="unsupervised_test",
    )

    # No target column for clustering.

    target_column = None

    # --------------------------------------------------------
    # TRAINING DATA PREVIEW
    # --------------------------------------------------------

    if train_file is not None:

        try:

            train_preview = pd.read_csv(
                train_file
            )

            st.write(
                f"Training dataset: "
                f"{train_preview.shape[0]} rows × "
                f"{train_preview.shape[1]} columns"
            )

        except Exception as error:

            st.error(
                f"Could not read the training CSV: {error}"
            )

            train_preview = None


# ============================================================
# DETECT CURRENT INPUT
# ============================================================

current_input_signature = (
    build_input_signature(
        learning_type,
        train_file,
        test_file,
        target_column,
    )
)


previous_signature = (
    st.session_state.input_signature
)


# ============================================================
# RESET OLD RESULTS WHEN INPUT CHANGES
# ============================================================

if (
    previous_signature is not None
    and current_input_signature
    != previous_signature
    and not st.session_state.training_running
):

    st.session_state.result = None

    st.session_state.analysis_complete = False

    st.session_state.cancel_message = None


# ============================================================
# INPUT READINESS
# ============================================================

if learning_type == "Supervised":

    ready = (
        train_file is not None
        and test_file is not None
        and target_column is not None
        and train_preview is not None
    )

else:

    ready = (
        train_file is not None
        and test_file is not None
        and train_preview is not None
    )


# ============================================================
# FIND BEST MODEL
# ============================================================

st.divider()

st.subheader(
    "3. Find Best Model"
)

st.warning(
    "⏳ Finding the best model is a complex operation "
    "involving multiple models and cross-validation. "
    "It may take around 5–10 minutes."
)


# ============================================================
# FIND BUTTON
# ============================================================

find_disabled = (
    not ready
    or st.session_state.training_running
    or st.session_state.analysis_complete
)


find_model = st.button(
    "🚀 Find Best Model",
    type="primary",
    use_container_width=True,
    disabled=find_disabled,
)


# ============================================================
# INPUT REQUIREMENTS
# ============================================================

if not ready:

    if learning_type == "Supervised":

        st.caption(
            "Upload both training and testing datasets "
            "and select a target column to continue."
        )

    else:

        st.caption(
            "Upload both training and testing datasets "
            "to continue."
        )


# ============================================================
# START TRAINING
# ============================================================

if find_model:

    try:

        job_id = start_training(
            train_file,
            test_file,
            learning_type,
            target_column,
        )

        st.session_state.job_id = job_id

        st.session_state.training_running = True

        st.session_state.analysis_complete = False

        st.session_state.result = None

        st.session_state.input_signature = (
            current_input_signature
        )

        st.session_state.cancel_message = None

        st.rerun()

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI backend. "
            "Make sure the backend is running on "
            "http://127.0.0.1:8000."
        )

    except requests.exceptions.Timeout:

        st.error(
            "❌ The backend took too long to respond "
            "while starting the training job."
        )

    except Exception as error:

        st.error(
            f"❌ Could not start model training: {error}"
        )


# ============================================================
# RUNNING TRAINING UI
# ============================================================

if st.session_state.training_running:

    job_id = st.session_state.job_id

    st.warning(
        "🔄 Model training is currently running..."
    )

    st.info(
        "The system is training and comparing multiple "
        "models. Please wait."
    )

    # --------------------------------------------------------
    # CANCEL BUTTON
    # --------------------------------------------------------

    if st.button(
        "🛑 Cancel Training",
        type="secondary",
        use_container_width=True,
        key="cancel_training_button",
    ):

        try:

            cancel_response = (
                cancel_training(
                    job_id
                )
            )

            if (
                cancel_response.get(
                    "status"
                )
                == "cancelled"
            ):

                st.session_state.training_running = False

                st.session_state.job_id = None

                st.session_state.result = None

                st.session_state.analysis_complete = False

                st.session_state.cancel_message = (
                    "Training was cancelled."
                )

                st.rerun()

            else:

                st.warning(
                    cancel_response.get(
                        "message",
                        "Could not cancel training.",
                    )
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the FastAPI "
                "backend to cancel training."
            )

        except Exception as error:

            st.error(
                f"❌ Could not cancel training: {error}"
            )

    # --------------------------------------------------------
    # CHECK TRAINING STATUS
    # --------------------------------------------------------

    else:

        try:

            status_response = (
                get_training_status(
                    job_id
                )
            )

            status = status_response.get(
                "status"
            )

            # =================================================
            # RUNNING
            # =================================================

            if status == "running":

                time.sleep(1)

                st.rerun()

            # =================================================
            # COMPLETED
            # =================================================

            elif status == "completed":

                result = status_response.get(
                    "result"
                )

                if result is None:

                    st.session_state.training_running = False

                    st.session_state.job_id = None

                    st.session_state.analysis_complete = False

                    st.error(
                        "❌ Training completed but "
                        "no result was returned."
                    )

                else:

                    st.session_state.result = result

                    st.session_state.training_running = False

                    st.session_state.job_id = None

                    st.session_state.analysis_complete = True

                    st.session_state.input_signature = (
                        current_input_signature
                    )

                    st.session_state.cancel_message = None

                    st.rerun()

            # =================================================
            # CANCELLED
            # =================================================

            elif status == "cancelled":

                st.session_state.training_running = False

                st.session_state.job_id = None

                st.session_state.result = None

                st.session_state.analysis_complete = False

                st.session_state.cancel_message = (
                    "Training was cancelled."
                )

                st.rerun()

            # =================================================
            # FAILED
            # =================================================

            elif status == "failed":

                st.session_state.training_running = False

                st.session_state.job_id = None

                st.session_state.analysis_complete = False

                detail = status_response.get(
                    "detail",
                    "Training failed.",
                )

                st.error(
                    f"❌ Model training failed: {detail}"
                )

            # =================================================
            # UNKNOWN
            # =================================================

            else:

                st.session_state.training_running = False

                st.session_state.job_id = None

                st.session_state.analysis_complete = False

                st.error(
                    f"❌ Unknown training status: {status}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Lost connection to the FastAPI backend "
                "while training was running."
            )

        except requests.exceptions.Timeout:

            st.warning(
                "⚠️ Could not check the training status. "
                "Retrying..."
            )

            time.sleep(1)

            st.rerun()

        except Exception as error:

            st.session_state.training_running = False

            st.session_state.job_id = None

            st.session_state.analysis_complete = False

            st.error(
                f"❌ Could not check training status: {error}"
            )


# ============================================================
# CANCEL MESSAGE
# ============================================================

if st.session_state.cancel_message:

    st.info(
        f"🛑 {st.session_state.cancel_message}"
    )


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result


if result is not None:

    st.success(
        "✅ Model analysis completed successfully!"
    )

    st.divider()

    # ========================================================
    # ANALYSIS INFORMATION
    # ========================================================

    st.subheader(
        "📊 Analysis"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Learning Type",
        result.get(
            "learning_type",
            learning_type,
        ).title(),
    )

    current_task = result.get(
        "task",
        "Unknown",
    ).lower()

    col2.metric(
        "Task",
        current_task.title(),
    )

    validation = result.get(
        "validation",
        {},
    )

    col3.metric(
        "Features",
        validation.get(
            "n_features",
            "N/A",
        ),
    )

    col4.metric(
        "Training Rows",
        validation.get(
            "n_train_rows",
            "N/A",
        ),
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader(
        "🏆 Best Model"
    )

    best_model = result.get(
        "best_model",
        "Unknown",
    )

    st.markdown(
        f"### {best_model}"
    )


    # ========================================================
    # CLUSTERING
    #
    # ONLY:
    # - Best Number of Clusters
    # - Silhouette Score
    #
    # Davies-Bouldin and Calinski-Harabasz are NOT displayed.
    # ========================================================

    if current_task == "clustering":

        best_k = result.get(
            "best_k"
        )

        best_scores = result.get(
            "best_model_scores",
            {},
        )

        silhouette = best_scores.get(
            "silhouette"
        )

        if silhouette is None:

            silhouette = best_scores.get(
                "silhouette_score"
            )

        cluster_col1, cluster_col2 = (
            st.columns(2)
        )

        with cluster_col1:

            st.metric(
                "Best Number of Clusters",
                (
                    best_k
                    if best_k is not None
                    else "N/A"
                ),
            )

        with cluster_col2:

            if silhouette is None:

                silhouette_display = "N/A"

            else:

                # Backend already returns the silhouette
                # on the 0–100 scale.

                silhouette_display = (
                    f"{float(silhouette):.2f}"
                )

            st.metric(
                "Silhouette Score",
                silhouette_display,
            )


    # ========================================================
    # NON-CLUSTERING METRICS
    # ========================================================

    elif current_task != "clustering":

        best_scores = result.get(
            "best_model_scores",
            {},
        )

        if best_scores:

            score_columns = st.columns(
                len(best_scores)
            )

            for column, (
                metric,
                value,
            ) in zip(
                score_columns,
                best_scores.items(),
            ):

                if value is None:

                    display_value = "N/A"

                elif isinstance(
                    value,
                    (int, float),
                ):

                    if current_task == "regression":

                        if metric.lower() in [
                            "r2",
                            "r2_score",
                            "r²",
                        ]:

                            display_value = (
                                f"{value:.2f}%"
                            )

                        else:

                            display_value = (
                                f"{value:.2f}"
                            )

                    elif current_task == "classification":

                        display_value = (
                            f"{value:.2f}%"
                        )

                    else:

                        display_value = (
                            f"{value:.2f}"
                        )

                else:

                    display_value = str(
                        value
                    )

                column.metric(
                    metric.replace(
                        "_",
                        " ",
                    ).title(),
                    display_value,
                )


    # ========================================================
    # HYPERPARAMETERS
    # ========================================================

    hyperparameters = result.get(
        "best_hyperparameters",
        {},
    )

    if hyperparameters:

        st.subheader(
            "⚙️ Best Hyperparameters"
        )

        st.json(
            hyperparameters
        )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    all_models = result.get(
        "all_models",
        [],
    )

    if all_models:

        st.subheader(
            "📋 Model Comparison"
        )

        comparison_df = pd.DataFrame(
            all_models
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # CLUSTERING MODEL COMPARISON
    #
    # Keep this separate as a fallback for responses that
    # provide clustering results through "results".
    # ========================================================

    elif current_task == "clustering":

        clustering_results = result.get(
            "results",
            [],
        )

        if clustering_results:

            st.subheader(
                "📋 Clustering Model Comparison"
            )

            clustering_df = pd.DataFrame(
                clustering_results
            )

            st.dataframe(
                clustering_df,
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # MODEL SAVING
    # ========================================================

    model_saved = result.get(
        "model_saved",
        False,
    )

    metadata_saved = result.get(
        "metadata_saved",
        False,
    )

    if model_saved and metadata_saved:

        st.success(
            "✅ Best model and metadata "
            "were saved successfully."
        )

    elif model_saved:

        st.success(
            "✅ Best model was saved successfully."
        )

    elif metadata_saved:

        st.success(
            "✅ Model metadata was saved successfully."
        )


    # ========================================================
    # DOWNLOAD TRAINED MODEL
    #
    # The button appears ONLY after successful training
    # and only when the backend confirms that the model
    # was saved.
    # ========================================================

    if model_saved:

        st.subheader(
            "📦 Trained Model"
        )

        st.write(
            "Download the trained best model as a "
            "`.pkl` file for later use."
        )

        try:

            model_bytes = get_model_download(
                current_task
            )

            model_filename = (
                f"{current_task}_model.pkl"
            )

            st.download_button(
                label=(
                    "📥 Download Trained "
                    "Model (.pkl)"
                ),
                data=model_bytes,
                file_name=model_filename,
                mime="application/octet-stream",
                use_container_width=True,
                key=(
                    f"download_{current_task}"
                ),
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the FastAPI "
                "backend to prepare the model download."
            )

        except requests.exceptions.Timeout:

            st.error(
                "❌ The backend took too long to "
                "prepare the model download."
            )

        except Exception as error:

            st.error(
                f"❌ Could not prepare the model download: "
                f"{error}"
            )


    # ========================================================
    # SAVED MODEL INFORMATION
    # ========================================================

    model_path = result.get(
        "model_path"
    )

    metadata_path = result.get(
        "metadata_path"
    )

    if model_path:

        with st.expander(
            "📁 Saved Model Information"
        ):

            st.write(
                "**Model:**"
            )

            st.code(
                model_path
            )

            if metadata_path:

                st.write(
                    "**Metadata:**"
                )

                st.code(
                    metadata_path
                )


    # ========================================================
    # START NEW ANALYSIS
    # ========================================================

    st.divider()

    if st.button(
        "🔄 Start New Analysis",
        use_container_width=True,
    ):

        clear_analysis()

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Automation • V1"
)