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

    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}@{alias}"

    model = mlflow.pyfunc.load_model(model_uri)

    df = load_data(data_path)
    if limit:
        df = df.head(limit)
    X, y, feature_names = preprocess(df)

    if model:
        predictions = model.predict(X)
        result_df = pd.DataFrame({"actual": y, "predicted": predictions})

        if output:
            result_df.to_csv(output, index=False)
            print(f"Predictions written to {output}")
        else:
            print(result_df.head(20))
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
