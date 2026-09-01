#!/usr/bin/env python
"""
Load a registered model and score orders data.
TASK: Load the registered model using the appropriate URI (alias or version).
"""

import argparse
import os
from typing import Optional

import pandas as pd
import mlflow.pyfunc

from src.data_preprocessing import load_data, preprocess


def make_predictions(
    model_name: str = "MLflowModel",
    alias: Optional[str] = "champion",
    version: Optional[int] = None,
    data_path: Optional[str] = None,
    output: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Loads a model and generates predictions.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # -------------------------------------------------------------------------
    # TASK: Implement Model Loading
    #
    # 1. Construct the model URI using the provided 'model_name'.
    #    - If 'version' is provided, the URI should use the specific version.
    #    - Otherwise, use the 'alias' provided.
    # 2. Use the MLflow 'pyfunc' flavour to load the model from the URI.
    # -------------------------------------------------------------------------

    # TODO: Implement model loading logic here
    model = None

    df = load_data(data_path)
    if limit:
        df = df.head(limit)
    X, y, feature_names = preprocess(df)

    if model:
        # -------------------------------------------------------------------------
        # TASK: Generate Predictions
        # 1. Use the loaded 'model' to predict values from the processed data 'X'.
        # 2. Create a DataFrame with columns "actual" and "predicted" containing true and predicted values.
        # 3. If an output file path is provided, write the DataFrame to CSV and notify the user.
        #    Otherwise, print the top 20 rows of the DataFrame to the console.
        # -------------------------------------------------------------------------

        # TODO: Implement prediction logic here
    else:
        print("Model not loaded in make_predictions(). Implement loading logic.")

    print("Prediction task complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="MLflowModel")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--alias", default="champion")
    group.add_argument("--version", type=int, default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    alias_val = args.alias if args.version is None else None

    make_predictions(
        model_name=args.model_name,
        alias=alias_val,
        version=args.version,
        data_path=args.data_path,
        output=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
