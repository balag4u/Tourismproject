# for standard library utilities
import os
import time

# for data manipulation
import pandas as pd

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

# for experiment tracking
import mlflow

# for model serialization
import joblib

# for Hugging Face model repository interaction
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Load dataset
dataset = pd.read_csv("tourismproject/data/tourism.csv")

target = "ProdTaken"

# Feature definitions
numeric_features = [
    "Age",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]

categorical_features = [
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]

binary_features = ["Passport", "OwnCar"]

X = dataset[numeric_features + categorical_features + binary_features]
y = dataset[target]

# Train–test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="passthrough",
)

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("travel-package-models")

classification_threshold = 0.45

# Global best model tracking
best_global_f1 = -1
best_global_model = None
best_global_name = None

# Model configurations and hyperparameter grids
model_configs = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "param_grid": {
            "decisiontreeclassifier__max_depth": [3, 5, 7],
            "decisiontreeclassifier__min_samples_split": [2, 5, 10],
            "decisiontreeclassifier__criterion": ["gini", "entropy"],
        },
    },
    "Bagging": {
        "model": BaggingClassifier(random_state=42),
        "param_grid": {
            "baggingclassifier__n_estimators": [30, 50, 70],
            "baggingclassifier__max_samples": [0.5, 0.7, 1.0],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "param_grid": {
            "randomforestclassifier__n_estimators": [100, 150],
            "randomforestclassifier__max_depth": [5, 7],
            "randomforestclassifier__min_samples_split": [2, 5],
        },
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "param_grid": {
            "adaboostclassifier__n_estimators": [50, 100],
            "adaboostclassifier__learning_rate": [0.01, 0.1],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "gradientboostingclassifier__n_estimators": [50, 100],
            "gradientboostingclassifier__learning_rate": [0.05, 0.1],
            "gradientboostingclassifier__max_depth": [2, 3],
        },
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=class_weight,
            eval_metric="logloss",
        ),
        "param_grid": {
            "xgbclassifier__n_estimators": [100, 150],
            "xgbclassifier__learning_rate": [0.05, 0.1],
            "xgbclassifier__max_depth": [2, 3],
            "xgbclassifier__colsample_bytree": [0.6, 0.8],
        },
    },
}

# Training, tuning, evaluation, and MLflow logging
def train_tune_log(model_name, model_obj, param_grid):
    global best_global_f1, best_global_model, best_global_name

    with mlflow.start_run(run_name=model_name):

        pipeline = make_pipeline(preprocessor, model_obj)

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            n_jobs=-1,
            scoring="f1",
            verbose=0,
        )

        grid.fit(Xtrain, ytrain)

        results = grid.cv_results_

        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
                mlflow.log_metric("std_test_score", results["std_test_score"][i])
            time.sleep(0.5)

        best_model = grid.best_estimator_

        y_train_pred = (
            best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold
        ).astype(int)

        y_test_pred = (
            best_model.predict_proba(Xtest)[:, 1] >= classification_threshold
        ).astype(int)

        train_report = classification_report(
            ytrain, y_train_pred, output_dict=True
        )
        test_report = classification_report(
            ytest, y_test_pred, output_dict=True
        )

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(
            {
                "train_accuracy": train_report["accuracy"],
                "train_recall": train_report["1"]["recall"],
                "train_precision": train_report["1"]["precision"],
                "train_f1": train_report["1"]["f1-score"],
                "test_accuracy": test_report["accuracy"],
                "test_recall": test_report["1"]["recall"],
                "test_precision": test_report["1"]["precision"],
                "test_f1": test_report["1"]["f1-score"],
            }
        )

        model_path = f"{model_name}_best_model.joblib"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        if test_report["1"]["f1-score"] > best_global_f1:
            best_global_f1 = test_report["1"]["f1-score"]
            best_global_model = best_model
            best_global_name = model_name

# Run training for all models
for model_name, cfg in model_configs.items():
    train_tune_log(model_name, cfg["model"], cfg["param_grid"])

# Save and upload the best global model to Hugging Face
final_model_path = "best_tourism_model.joblib"
joblib.dump(best_global_model, final_model_path)

api = HfApi()
repo_id = "nsa9/tourism-best-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=final_model_path,
    path_in_repo=final_model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
