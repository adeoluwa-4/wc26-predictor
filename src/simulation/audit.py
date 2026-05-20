"""Simulation input diagnostics and audit artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.config import TRAINING_TABLE_PATH
from src.utils.paths import PROJECT_ROOT


ELITE_TEAMS = ("Brazil", "France", "Argentina", "Spain", "England", "Portugal", "Netherlands", "Germany")


def _table_to_records(df: pd.DataFrame, cols: list[str], n: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df[cols].head(n).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out.to_dict(orient="records")


def _read_table_with_csv_fallback(parquet_path: Path) -> pd.DataFrame:
    """Read parquet when available; fall back to CSV with same stem."""
    csv_path = parquet_path.with_suffix(".csv")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing table: {parquet_path} or {csv_path}")


def build_strength_sanity() -> dict[str, Any]:
    """Build sanity tables from processed training data and model profiles."""
    warnings: list[str] = []
    tables: dict[str, Any] = {
        "top_25_recent_elo": [],
        "top_25_recent_fifa_points": [],
        "top_25_model_implied_strength": [],
        "warnings": warnings,
    }

    training_path = Path(TRAINING_TABLE_PATH)
    if not training_path.exists():
        warnings.append(f"Training table not found at {training_path}")
        return tables

    try:
        df = _read_table_with_csv_fallback(training_path)
    except Exception as exc:
        warnings.append(f"Failed reading training table {training_path}: {type(exc).__name__}: {exc}")
        return tables
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    recent = df[df["date"] >= pd.Timestamp("2018-01-01")].copy()
    if recent.empty:
        warnings.append("No rows found since 2018 for strength sanity tables")
        return tables

    home = recent[["home_team", "home_elo", "home_fifa_points"]].rename(
        columns={"home_team": "team", "home_elo": "elo", "home_fifa_points": "fifa_points"}
    )
    away = recent[["away_team", "away_elo", "away_fifa_points"]].rename(
        columns={"away_team": "team", "away_elo": "elo", "away_fifa_points": "fifa_points"}
    )
    long_df = pd.concat([home, away], ignore_index=True)

    top_elo = (
        long_df.dropna(subset=["elo"])
        .groupby("team", as_index=False)["elo"]
        .mean()
        .sort_values("elo", ascending=False)
    )
    top_fifa = (
        long_df.dropna(subset=["fifa_points"])
        .groupby("team", as_index=False)["fifa_points"]
        .mean()
        .sort_values("fifa_points", ascending=False)
    )

    tables["top_25_recent_elo"] = _table_to_records(top_elo, ["team", "elo"], 25)
    tables["top_25_recent_fifa_points"] = _table_to_records(top_fifa, ["team", "fifa_points"], 25)

    missing_elite = sorted(set(ELITE_TEAMS) - set(top_elo["team"].head(20).tolist()))
    if missing_elite:
        warnings.append(f"Elite teams missing from top-20 recent Elo table: {missing_elite}")

    model_profiles = PROJECT_ROOT / "models" / "team_profiles.parquet"
    if model_profiles.exists() or model_profiles.with_suffix(".csv").exists():
        try:
            profiles = _read_table_with_csv_fallback(model_profiles)
        except Exception as exc:
            warnings.append(f"Failed reading model profiles {model_profiles}: {type(exc).__name__}: {exc}")
            return tables
        profiles["implied_strength"] = (
            0.7 * profiles["elo"].fillna(profiles["elo"].median(skipna=True))
            + 0.3 * profiles["fifa_points"].fillna(profiles["fifa_points"].median(skipna=True))
        )
        implied = profiles.sort_values("implied_strength", ascending=False)
        tables["top_25_model_implied_strength"] = _table_to_records(
            implied,
            ["team", "implied_strength", "elo", "fifa_points"],
            25,
        )
    else:
        warnings.append(f"Model profile file not found at {model_profiles}")

    return tables


def write_simulation_input_audit(
    output_path: str | Path,
    groups: dict[str, list[str]],
    team_config: pd.DataFrame,
) -> dict[str, Any]:
    """Write simulation input audit JSON with group source and sanity tables."""
    strength = build_strength_sanity()
    projected = team_config[team_config["status"] == "projected_placeholder"].copy()

    artifact: dict[str, Any] = {
        "groups_used": groups,
        "slot_status_counts": team_config["status"].value_counts().to_dict(),
        "projected_placeholders": projected[
            ["group", "team", "status", "source", "notes"]
        ].to_dict(orient="records"),
        "top_20_recent_elo": strength["top_25_recent_elo"][:20],
        "top_25_recent_elo": strength["top_25_recent_elo"],
        "top_25_recent_fifa_points": strength["top_25_recent_fifa_points"],
        "top_25_model_implied_strength": strength["top_25_model_implied_strength"],
        "feature_sanity_warnings": strength["warnings"],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact
