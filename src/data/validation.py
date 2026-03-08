"""Validation checks for the final training table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def expected_output_columns(windows: Sequence[int]) -> list[str]:
    """Return expected final output columns for configured rolling windows."""
    cols = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "tournament",
        "city",
        "country",
        "neutral",
        "home_score",
        "away_score",
        "home_win",
        "away_win",
        "draw",
        "outcome_label",
        "total_goals",
        "goal_diff",
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_fifa_rank",
        "away_fifa_rank",
        "fifa_rank_diff",
        "home_fifa_points",
        "away_fifa_points",
        "fifa_points_diff",
        "tournament_type",
        "is_friendly",
        "is_qualifier",
        "is_continental_competition",
        "is_world_cup",
        "is_host_home_country",
        "is_host_away_country",
        "same_confederation",
        "confederation_home",
        "confederation_away",
        "h2h_matches_prior",
        "h2h_home_team_wins_prior",
        "h2h_away_team_wins_prior",
        "h2h_draws_prior",
        "h2h_goal_diff_prior",
        "went_to_shootout",
        "shootout_winner",
    ]

    metrics = ("points", "points_per_match", "goals_for", "goals_against", "goal_diff", "wins", "draws", "losses")
    for window in windows:
        for prefix in ("home", "away"):
            for metric in metrics:
                cols.append(f"{prefix}_{metric}_last_{window}")

    return cols


def validate_training_table(
    df: pd.DataFrame,
    windows: Sequence[int],
    join_report: dict[str, float] | None = None,
) -> dict[str, object]:
    """Run data quality and leakage-safety checks."""
    join_report = join_report or {}

    errors: list[str] = []
    warnings: list[str] = []

    expected_cols = set(expected_output_columns(windows))
    actual_cols = set(df.columns)

    missing_cols = sorted(expected_cols - actual_cols)
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    duplicate_match_ids = int(df["match_id"].duplicated().sum())
    if duplicate_match_ids > 0:
        errors.append(f"Found duplicate match_id values: {duplicate_match_ids}")

    if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
        errors.append("Negative score values found")

    invalid_outcome_rows = int(
        (
            (df["home_win"] + df["away_win"] + df["draw"] != 1)
            | ((df["outcome_label"] == "home_win") & (df["home_win"] != 1))
            | ((df["outcome_label"] == "away_win") & (df["away_win"] != 1))
            | ((df["outcome_label"] == "draw") & (df["draw"] != 1))
        ).sum()
    )
    if invalid_outcome_rows > 0:
        errors.append(f"Outcome target inconsistency in {invalid_outcome_rows} rows")

    if df["date"].isna().any():
        errors.append("Found null date values")

    if not df["date"].is_monotonic_increasing:
        warnings.append("Final table is not globally sorted by date")

    # Rolling sanity checks (bounds).
    for window in windows:
        for side in ("home", "away"):
            wins_col = f"{side}_wins_last_{window}"
            draws_col = f"{side}_draws_last_{window}"
            losses_col = f"{side}_losses_last_{window}"
            points_col = f"{side}_points_last_{window}"

            invalid_counts = (
                (df[wins_col] < 0)
                | (df[draws_col] < 0)
                | (df[losses_col] < 0)
                | (df[points_col] < 0)
                | (df[wins_col] > window)
                | (df[draws_col] > window)
                | (df[losses_col] > window)
                | (df[points_col] > 3 * window)
                | (df[wins_col] + df[draws_col] + df[losses_col] > window)
            )
            bad_rows = int(invalid_counts.sum())
            if bad_rows > 0:
                errors.append(f"Rolling bounds violated in {bad_rows} rows for {side} window {window}")

    non_nullable = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "away_win",
        "draw",
        "outcome_label",
        "h2h_matches_prior",
        "h2h_home_team_wins_prior",
        "h2h_away_team_wins_prior",
        "h2h_draws_prior",
        "h2h_goal_diff_prior",
        "went_to_shootout",
    ]

    for col in non_nullable:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append(f"Column {col} has unexpected nulls: {null_count}")

    # Rankings may be sparse in very old years, so we warn rather than fail.
    for metric, coverage in join_report.items():
        if coverage < 0.30:
            warnings.append(f"Low join coverage for {metric}: {coverage:.3f}")

    report = {
        "status": "pass" if not errors else "fail",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "errors": errors,
        "warnings": warnings,
        "join_coverage": join_report,
    }
    return report


def write_validation_report(report: dict[str, object], path: Path) -> None:
    """Write validation report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
