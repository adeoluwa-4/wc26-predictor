"""Read-only dashboard access to DuckDB analytics marts and published artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.models.config import WAREHOUSE_PATH
from src.utils.paths import PROJECT_ROOT


LATEST_PREDICTION_PATH = PROJECT_ROOT / "data" / "published" / "latest.json"


def read_mart(relation: str, warehouse_path: Path = WAREHOUSE_PATH) -> pd.DataFrame:
    allowed = {
        "dim_team",
        "dim_tournament",
        "dim_date",
        "fact_match",
        "fact_team_ranking",
        "fact_team_form",
        "mart_team_performance",
        "mart_model_features",
        "mart_predictions",
    }
    if relation not in allowed:
        raise ValueError(f"Unsupported analytics relation: {relation}")
    if not warehouse_path.exists():
        return pd.DataFrame()
    connection = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        return connection.execute(f"select * from analytics.{relation}").df()
    except duckdb.Error:
        return pd.DataFrame()
    finally:
        connection.close()


def published_prediction_metadata(path: Path = LATEST_PREDICTION_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
