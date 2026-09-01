# Practical Task: MLflow Lifecycle Management

This practical task is designed to teach the end-to-end Machine Learning lifecycle using MLflow. It covers everything from experimental training and logging to model registration, inference, and containerized deployment readiness.

## Setup
To get started, you need to set up a local Python environment and install the necessary dependencies. This ensures that all components of the MLflow pipeline run consistently on your machine.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Detailed Task Description
The primary goal is to complete the task scripts in `src/` and understand how each stage of the MLflow lifecycle fits together for the **orders** dataset.

1. **Training & Logging (`src/train.py`)**
   - Train either `logistic` or `rf` model on preprocessed orders data.
   - Create/select an MLflow experiment.
   - Log model params, evaluation metrics (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`), model artifact, and diagnostic plots.
2. **Model Registry (`src/register_model.py`)**
   - Find the best run for a selected metric (default `f1_score`).
   - Register the run's model artifact into MLflow Model Registry.
   - Assign alias (default `champion`) to the registered version.
3. **Inference (`src/predict.py`)**
   - Load model by alias (`models:/<name>@champion`) or explicit version (`models:/<name>/<version>`).
   - Run prediction on preprocessed input data.
   - Save results to CSV with `predicted` and, when available, `actual`.
4. **Containerization (`src/build_image.py`)**
   - Build a serving Docker image from a registered model URI using MLflow model tooling.

## Usage
The project is configured with an `MLproject` file, allowing you to run different stages of the pipeline using the `mlflow run` command. Ensure your environment is activated before running these.

```bash
# 1. Train a model (supports 'rf' or 'logistic')
mlflow run . -e train --env-manager=local

# 2. Register the best model from the experiment
mlflow run . -e register --env-manager=local

# 3. Generate predictions using the registered model
mlflow run . -e predict --env-manager=local

# 4. Build a Docker image for the model
mlflow run . -e build_image --env-manager=local
```

## Tests
A suite of unit and integration tests verifies each lifecycle stage. Run tests frequently while implementing task files.

```bash
# Run all tests with verbose output
pytest tests/ -v
```

Key test files:
- `tests/test_train.py`: validates training run creation and logging.
- `tests/test_register.py`: validates model registration and alias assignment.
- `tests/test_predict.py`: validates loading registered model and producing predictions.
- `tests/test_pipeline.py`: end-to-end flow (train -> register -> predict -> optional Docker build).

## Dataset
The pipeline is configured for the orders dataset at `data/orders.csv`.

- **Dataset file**: `data/orders.csv`
- **Data dictionary**: `datadictionary.md` (column meanings, value domains, missingness, ranges)
- **Preprocessing entrypoint**: `src/data_preprocessing.py`
- **Target definition**: binary label derived from `status` (`delivered` -> `1`, all others -> `0`)
