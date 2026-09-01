"""Tests for model registration."""
import os
import mlflow
import pytest
from src.train import run_training
from src.register_model import register_best_model

def test_register_model_assigns_alias(tmp_db_uri, sample_csv_path):
    """
    TASK: Implement register_best_model in src.register_model.
    The function should find the best run and register it with an alias.
    """
    os.environ["MLFLOW_TRACKING_URI"] = tmp_db_uri
    mlflow.set_tracking_uri(tmp_db_uri)
    
    # 1. Train a model first to have something to register
    run_training(
        model_type="logistic",
        data_path=sample_csv_path,
        experiment_name="mlflow-pipeline"
    )

    # 2. Register the model
    register_best_model(
        experiment_name="mlflow-pipeline",
        metric="f1_score",
        model_name="MLflowModel",
        alias="champion"
    )

    # 3. Verify registration with MLflow Client
    client = mlflow.tracking.MlflowClient()
    try:
        registered_model = client.get_registered_model("MLflowModel")
        assert registered_model is not None, "Model 'MLflowModel' was not registered."
        
        # Check for the alias
        # Note: Aliases are stored on model versions, but can be retrieved via the client
        # or by checking the latest version's aliases attribute
        found_alias = False
        for version in registered_model.latest_versions:
             # Some versions of MLflow might store aliases differently, 
             # but set_registered_model_alias is the standard way.
             aliases = client.get_model_version_by_alias("MLflowModel", "champion")
             if aliases:
                 found_alias = True
                 break
        
        assert found_alias, "Alias 'champion' was not assigned to any model version."
    except Exception as e:
        pytest.fail(f"Verification of registration failed: {e}")
