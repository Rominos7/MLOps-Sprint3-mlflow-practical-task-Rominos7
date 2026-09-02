#!/usr/bin/env python
"""
Register the best model from an experiment.
TASK: Implement MLflow model registration and aliasing.
"""

import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient


def register_best_model(
    experiment_name: str = "mlflow-pipeline",
    metric: str = "f1_score",
    model_name: str = "MLflowModel",
    alias: str = "champion",
) -> None:
    """
    Finds the best run in an experiment and registers it.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")

    best_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=alias, version=result.version)

    print(f"Model version {result.version} registered as '{model_name}' with alias '{alias}'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="mlflow-pipeline")
    parser.add_argument("--metric", default="f1_score")
    parser.add_argument("--model-name", default="MLflowModel")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    register_best_model(
        experiment_name=args.experiment_name,
        metric=args.metric,
        model_name=args.model_name,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()
