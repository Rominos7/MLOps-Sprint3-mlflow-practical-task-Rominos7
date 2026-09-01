"""Tests for training pipeline."""
import os

import mlflow
import pytest

from src.train import run_training


def test_train_runs_and_logs_to_mlflow(tmp_db_uri, sample_csv_path):
    """
    TASK: Implement logging in src.train.run_training.
    The function should complete and create an MLflow run in the experiment 'mlflow-pipeline'.
    """
    os.environ["MLFLOW_TRACKING_URI"] = tmp_db_uri
    
    # Run training directly
    run_training(
        model_type="logistic",
        data_path=sample_csv_path,
        experiment_name="mlflow-pipeline"
    )

    mlflow.set_tracking_uri(tmp_db_uri)
    exp = mlflow.get_experiment_by_name("mlflow-pipeline")
    assert exp is not None, "Experiment 'mlflow-pipeline' not found."
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 1, "No MLflow runs found. Ensure logging is implemented."
    
    # Check if key metrics and params are logged
    # (These will fail initially for students)
    assert "metrics.f1_score" in runs.columns or "metrics.f1-score" in runs.columns, "f1_score metric not logged."
    assert "params.model_type" in runs.columns or "params.C" in runs.columns, "Model parameters not logged."
