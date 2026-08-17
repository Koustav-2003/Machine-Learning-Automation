# Machine Learning Automation

An automated machine learning application for **Regression, Classification, and Clustering**. The application trains multiple machine learning models, performs hyperparameter tuning, compares model performance, identifies the best-performing model, and allows the trained model to be downloaded.

## 🚀 Deployed Application

**Deployed App (Streamlit):**  
[https://machine-learning-automation.streamlit.app/](https://machine-learning-automation.streamlit.app/)

**Deployed Backend (FastAPI):**  
[https://machine-learning-automation.onrender.com](https://dashboard.render.com/web/srv-da0jlte7bikc73fa2830)

## 📓 Try the Notebook Versions

The notebooks below use the **Heavy Version** of the pipelines, with more extensive hyperparameter searches than the deployed version.

### Regression — Heavy Version

[Try the Regression Notebook](https://drive.google.com/file/d/1nkl4VkUlaIF9xRiOp_IzoH0BaYYjWEH7/view?usp=sharing)

### Classification — Heavy Version

[Try the Classification Notebook](https://drive.google.com/file/d/1oWluGDgX5l3wqQSkaLXcnjEvYZGlbAl_/view?usp=sharing)

### Clustering

[Try the Clustering Notebook](https://drive.google.com/file/d/1hueedosWSOHe45jVTEMErijXCv4K6z6A/view?usp=sharing)

---

# 📌 Dataset Requirement

The dataset uploaded to this application must already be **properly processed**.

The ML Automation application focuses on model training, comparison, evaluation, and selection rather than general-purpose data preprocessing.

You can use the companion preprocessing application:

**Deployed App:**  
[https://automated-ml-preprocessing.streamlit.app/](https://automated-ml-preprocessing.streamlit.app/)

**GitHub Repository:**  
[https://github.com/Koustav-2003/automated-ml-preprocessing](https://github.com/Koustav-2003/automated-ml-preprocessing)

### Recommended workflow

```text
Raw Dataset
     │
     ▼
Automated ML Preprocessing
     │
     ▼
Processed Dataset
     │
     ▼
Machine Learning Automation
     │
     ├── Regression
     ├── Classification
     └── Clustering
```

---

# 🏗️ Architecture

The application follows a **frontend → backend → ML pipeline** architecture.

```text
                    ┌─────────────────────────┐
                    │       User / Browser     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Streamlit Frontend    │
                    │         app.py           │
                    └────────────┬────────────┘
                                 │ HTTP Requests
                                 ▼
                    ┌─────────────────────────┐
                    │      FastAPI Backend    │
                    │        routes.py        │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Regression   │ │Classification│ │  Clustering  │
        │  Pipeline    │ │   Pipeline   │ │   Pipeline   │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               └────────────────┼────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ Model Comparison &      │
                    │ Best Model Selection    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             Results / Scores          Trained .pkl Model
```

### Main Components

**Frontend — Streamlit**

- Dataset upload
- Task selection
- Target-column selection where applicable
- Training controls
- Training status
- Model comparison
- Best-model display
- `.pkl` model download

**Backend — FastAPI**

- Training requests
- Regression execution
- Classification execution
- Clustering execution
- Training status
- Cancellation
- Result delivery
- Trained-model download

**ML Pipelines**

```text
backend/
│
├── routes.py
├── regression.py
├── classification.py
└── clustering.py
```

The API layer is separated from the machine-learning implementation.

---

# 🤖 Regression

The regression pipeline contains:

1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Support Vector Regression (SVR)
5. Decision Tree Regression
6. Random Forest Regression
7. XGBoost Regression
8. LightGBM Regression
9. CatBoost Regression

### Regression scores

- **R² Score**
- **MAE — Mean Absolute Error**
- **RMSE — Root Mean Squared Error**
- **Cross-Validation R²**

### Regression hyperparameters

| Model | Main parameters |
|---|---|
| Linear Regression | `fit_intercept`, `positive` |
| Polynomial Regression | `degree`, `fit_intercept` |
| SVR | `kernel`, `C`, `epsilon`, `gamma`, `degree`, `coef0` |
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| LightGBM | `n_estimators`, `learning_rate`, `num_leaves`, `max_depth` |
| CatBoost | `iterations`, `learning_rate`, `depth` |

---

# 🧠 Classification

The current classification pipeline contains:

1. Logistic Regression
2. K-Nearest Neighbours (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest
6. XGBoost
7. LightGBM
8. CatBoost

### Classification scores

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**
- **Cross-Validation Accuracy**

Precision, Recall, and F1 Score use weighted averaging.

ROC-AUC is calculated using predicted probabilities or decision scores when supported.

### Classification hyperparameters

| Model | Main parameters |
|---|---|
| Logistic Regression | `C`, `solver` |
| KNN | `n_neighbors`, `weights`, `metric` |
| SVM | `C`, `kernel`, `gamma` |
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| LightGBM | `n_estimators`, `learning_rate`, `num_leaves`, `max_depth` |
| CatBoost | `iterations`, `learning_rate`, `depth` |

---

# 🔵 Clustering

The clustering pipeline uses:

1. **K-Means Clustering**
2. **Hierarchical / Agglomerative Clustering**

Clustering is unsupervised, so there is no conventional target column or test-set accuracy.

### K-Means parameters

- `n_clusters`
- `init`
- `n_init`
- `max_iter`

### Hierarchical Clustering parameters

- `n_clusters`
- `linkage`

Supported linkage approaches include `ward`, `complete`, `average`, and `single`.

### Clustering evaluation

The main clustering quality measure is the **Silhouette Score**.

A higher Silhouette Score generally indicates better-separated and more cohesive clusters.

Candidate cluster counts are evaluated and the configuration with the best Silhouette Score is selected.

---

# ⚙️ Heavy Notebook vs Deployed Version

The project contains a more computationally intensive **Heavy Version** in the notebooks and a lighter **Lite Version** in the deployed application.

| Feature | Heavy Notebook | Deployed Version |
|---|---|---|
| Cross-validation | 5-fold | 2-fold |
| Search method | GridSearchCV | RandomizedSearchCV |
| Search strategy | Exhaustive combinations | Random sampled combinations |
| Hyperparameter space | Larger | Reduced |
| Search iterations | All combinations | 3 random combinations |
| Computational cost | Higher | Lower |
| Search coverage | Higher | Lower |

The Heavy version is **computationally heavy** because it evaluates substantially more model configurations and cross-validation folds.

The Heavy notebooks can therefore explore more combinations and may find better configurations, but higher search coverage does not guarantee a better test-set score.

---

# 🔄 How the Application Works

For supervised learning:

```text
Processed train.csv + test.csv
              │
              ▼
        Split X and y
              │
              ▼
    Cross-Validation / Tuning
              │
              ▼
       Train candidate models
              │
              ▼
     Select best configuration
              │
              ▼
   Fit best estimator on train data
              │
              ▼
      Evaluate on test.csv
              │
              ▼
       Compare model results
              │
              ▼
          Best Model
          /        \
         ▼          ▼
      Results     .pkl Download
```

`train.csv` is used for training, cross-validation, and hyperparameter selection. `test.csv` is reserved for final evaluation.

For clustering, the workflow is unsupervised and uses clustering quality metrics rather than supervised test accuracy.

---

# 📊 Model Selection

### Regression

Hyperparameters are selected using cross-validation R². The models are then compared using:

- R²
- MAE
- RMSE

### Classification

Hyperparameters are selected using cross-validation accuracy. The models are then compared using:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

### Clustering

Candidate clustering configurations are compared using:

- Silhouette Score
- Best number of clusters

---

# 💾 Trained Model Download

After an operation completes, the application provides an option to download the trained selected model as a `.pkl` file.

This allows the fitted estimator to be saved and reused without repeating the training process.

---

# 🛠️ Technology Stack

### Frontend

- Python
- Streamlit

### Backend

- Python
- FastAPI
- Uvicorn

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Data

- Pandas
- NumPy

### Model Persistence

- Joblib

---

# 📁 Project Structure

```text
Machine-Learning-Automation/
│
├── backend/
│   ├── routes.py
│   ├── regression.py
│   ├── classification.py
│   └── clustering.py
│
├── app.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

# 🌐 Deployment Architecture

```text
                    User
                     │
                     ▼
          Streamlit Community Cloud
                 Frontend
                (app.py)
                     │
                HTTP / API
                     │
                     ▼
                  Render
              FastAPI Backend
                (routes.py)
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     Regression Classification Clustering
       Pipeline     Pipeline    Pipeline
          │          │          │
          └──────────┼──────────┘
                     ▼
             Model Comparison
                     │
              ┌──────┴──────┐
              ▼             ▼
           Results      .pkl Model
```

---

# 📌 Important Notes

- Uploaded data must be properly processed before training.
- The companion preprocessing application can be used to prepare raw datasets.
- Regression and Classification use train/test datasets.
- Clustering is unsupervised and does not use conventional accuracy.
- The Heavy notebooks perform more extensive hyperparameter searches than the deployed Lite version.
- Model performance depends on dataset quality, feature representation, preprocessing, sample size, and target distribution.

---

# 📜 License

This project is licensed under the terms specified in the repository's `LICENSE` file.
