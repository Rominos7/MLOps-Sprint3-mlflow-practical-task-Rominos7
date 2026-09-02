#!/usr/bin/env python
"""
CLI training script.
TASK: Implement MLflow logging for the training process.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_preprocessing import load_data, preprocess, split_data

MODEL_REGISTRY = {
    "logistic": {
        "class": LogisticRegression,
        "defaults": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    },
    "rf": {
        "class": RandomForestClassifier,
        "defaults": {"n_estimators": 100, "max_depth": 10},
    },
}


def build_model(model_type: str, user_params: dict) -> tuple:
    entry = MODEL_REGISTRY[model_type]
    params = {**entry["defaults"], **user_params}
    params["random_state"] = params.get("random_state", 42)
    return entry["class"](**params), params


def evaluate(model: object, X_test: object, y_test: object) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_plots(model: object, X_test: object, y_test: object) -> None:
    """
    Helper to create diagnostic plots.
    TASK: Use MLflow to save these plots as artifacts in the active run.
    """
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm, cmap="Blues")
    fig_cm.tight_layout()
    mlflow.log_figure(fig_cm, "confusion_matrix.png")

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    fig_roc.tight_layout()
    mlflow.log_figure(fig_roc, "roc_curve.png")

    plt.close(fig_cm)
    plt.close(fig_roc)


def run_training(
    model_type: str = "rf",
    user_params: Dict[str, Any] = None,
    experiment_name: str = "mlflow-pipeline",
    data_path: Optional[str] = None,
) -> None:
    """
    Main training logic with MLflow logging.
    """
    if user_params is None:
        user_params = {}

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    df = load_data(data_path)
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model, params = build_model(model_type, user_params)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    mlflow.set_experiment(experiment_name)

    os.environ.pop("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_name=f"{model_type}_run"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test.iloc[:5],
        )
        log_plots(model, X_test, y_test)

    print(f"Training of {model_type} complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=list(MODEL_REGISTRY.keys()), default="rf")
    parser.add_argument("--params", type=str, default="{}")
    parser.add_argument("--experiment-name", default="mlflow-pipeline")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    user_params = json.loads(args.params)
    run_training(
        model_type=args.model_type,
        user_params=user_params,
        experiment_name=args.experiment_name,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
