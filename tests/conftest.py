"""Pytest fixtures. Synthetic data schema must match orders dataset."""
import os
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv_path(fixtures_dir):
    path = fixtures_dir / "sample_data.csv"
    if not path.exists():
        pytest.skip("sample_data.csv not found; create it to match your dataset schema")
    return str(path)


@pytest.fixture
def tmp_db_uri(tmp_path):
    """Temporary SQLite database URI for MLflow."""
    db_path = tmp_path / "mlflow.db"
    return f"sqlite:///{db_path}"
