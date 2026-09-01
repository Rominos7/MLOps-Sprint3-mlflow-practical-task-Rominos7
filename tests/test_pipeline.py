"""Integration test for the entire MLflow pipeline."""
import os
import mlflow
import pytest
import docker

from src.train import run_training
from src.register_model import register_best_model
from src.predict import make_predictions
from src.build_image import build_serving_image

def test_entire_pipeline_integration(tmp_db_uri, sample_csv_path, tmp_path):
    """
    Integration test: Runs the entire MLflow pipeline from training to Docker build.
    """
    os.environ["MLFLOW_TRACKING_URI"] = tmp_db_uri
    mlflow.set_tracking_uri(tmp_db_uri)
    
    # 1. Train the model
    run_training(
        model_type="logistic",
        data_path=sample_csv_path,
        experiment_name="mlflow-pipeline"
    )
    
    # Verify if run was created
    exp = mlflow.get_experiment_by_name("mlflow-pipeline")
    assert exp is not None, "Experiment 'mlflow-pipeline' not found."
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) > 0, "No MLflow runs found. Training logic not implemented."

    # 2. Register the model
    register_best_model(
        experiment_name="mlflow-pipeline",
        metric="f1_score",
        model_name="MLflowModel",
        alias="champion"
    )

    # Verify registration with MLflow API
    client = mlflow.MlflowClient()
    try:
        registered_model = client.get_registered_model("MLflowModel")
        # Check if version has 'champion' alias
        versions = client.get_latest_versions("MLflowModel")
        has_champion = False
        for v in versions:
             # Standard MLflow way to check aliases
             alias_version = client.get_model_version_by_alias("MLflowModel", "champion")
             if alias_version:
                 has_champion = True
                 break
        assert has_champion, "Model was not registered or 'champion' alias not found."
    except Exception as e:
        pytest.fail(f"Model registration verification failed: {e}")

    # 3. Generate predictions
    output_csv = tmp_path / "final_preds.csv"
    make_predictions(
        model_name="MLflowModel",
        alias="champion",
        data_path=sample_csv_path,
        output=str(output_csv),
        limit=2
    )
    assert output_csv.exists(), "Pipeline prediction output failed."

    # 4. Build Docker Image (Optional step depending on environment)
    docker_client = None
    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except Exception:
        pytest.skip("Docker daemon not available; skipping Docker build test step.")

    image_name = "mlflow-model-student-integration:latest"
    try:
        build_serving_image(
            model_name="MLflowModel",
            alias="champion",
            image_name=image_name
        )
        # Final Verification
        docker_client.images.get(image_name)
    except Exception as e:
        pytest.fail(f"Docker image building failed: {e}")
