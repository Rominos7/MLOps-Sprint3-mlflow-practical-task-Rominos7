"""Tests for prediction script."""
import os
import mlflow
import pytest
import pandas as pd
from src.train import run_training
from src.register_model import register_best_model
from src.predict import make_predictions

def test_make_predictions_generates_output(tmp_db_uri, sample_csv_path, tmp_path):
    """
    TASK: Implement make_predictions in src.predict.
    The function should load the champion model and generate predictions.
    """
    os.environ["MLFLOW_TRACKING_URI"] = tmp_db_uri
    mlflow.set_tracking_uri(tmp_db_uri)
    
    # 1. Setup: Train and Register
    run_training(
        model_type="logistic",
        data_path=sample_csv_path,
        experiment_name="mlflow-pipeline"
    )
    register_best_model(
        experiment_name="mlflow-pipeline",
        metric="f1_score",
        model_name="MLflowModel",
        alias="champion"
    )

    # 2. Run prediction
    output_csv = tmp_path / "preds.csv"
    make_predictions(
        model_name="MLflowModel",
        alias="champion",
        data_path=sample_csv_path,
        output=str(output_csv),
        limit=2
    )

    # 3. Verify output
    assert output_csv.exists(), "Prediction output file was not created."
    preds_df = pd.read_csv(output_csv)
    assert len(preds_df) == 2, "Prediction output row count mismatch."
    assert "actual" in preds_df.columns
    assert "predicted" in preds_df.columns
