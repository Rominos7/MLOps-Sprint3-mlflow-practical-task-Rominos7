#!/usr/bin/env python
"""
Build a Docker image for serving a registered model.
TASK: Implement Docker image building using the MLflow Python API.
"""

import argparse
import os
import mlflow.models


def build_serving_image(
    model_name: str = "MLflowModel",
    alias: str = "champion",
    image_name: str = "mlflow-model:latest",
) -> None:
    """
    Builds a Docker image using mlflow.models.build_docker.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}@{alias}"

    mlflow.models.build_docker(
        model_uri=model_uri,
        name=image_name,
        env_manager="local",
    )

    print(f"Docker image '{image_name}' built for model: {model_uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="MLflowModel")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--image-name", default="mlflow-model:latest")
    args = parser.parse_args()

    build_serving_image(
        model_name=args.model_name,
        alias=args.alias,
        image_name=args.image_name,
    )


if __name__ == "__main__":
    main()
