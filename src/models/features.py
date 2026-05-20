"""Feature selection and time-based splitting for modeling."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


ROLLING_PATTERN = re.compile(
    r"^(home|away)_(points|points_per_match|goals_for|goals_against|goal_diff|wins|draws|losses)_last_(5|10)$"
)

BASE_NUMERIC = [
    "neutral",
    "is_friendly",
    "is_qualifier",
    "is_continental_competition",
    "is_world_cup",
    "is_host_home_country",
    "is_host_away_country",
    "same_confederation",
    "tournament_importance_score",
    "home_elo",
    "away_elo",
    "elo_diff",
    "h2h_matches_prior",
    "h2h_home_team_wins_prior",
    "h2h_away_team_wins_prior",
    "h2h_draws_prior",
    "h2h_goal_diff_prior",
]

BASE_CATEGORICAL = [
    "home_team",
    "away_team",
    "tournament_type",
    "confederation_home",
    "confederation_away",
]


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end_date: pd.Timestamp
    val_end_date: pd.Timestamp


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric/categorical model feature columns present in the table."""
    rolling_cols = sorted(col for col in df.columns if ROLLING_PATTERN.match(col))
    numeric = [col for col in BASE_NUMERIC if col in df.columns] + rolling_cols
    categorical = [col for col in BASE_CATEGORICAL if col in df.columns]
    return numeric, categorical


def make_time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> TimeSplit:
    """Build strict chronological train/val/test split by unique dates."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    if "date" not in df.columns:
        raise ValueError("Input dataframe must contain date column")

    ordered = df.sort_values("date").reset_index(drop=True)
    unique_dates = ordered["date"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique dates for train/val/test split")

    train_idx = max(0, int(len(unique_dates) * train_frac) - 1)
    val_idx = max(train_idx + 1, int(len(unique_dates) * (train_frac + val_frac)) - 1)
    val_idx = min(val_idx, len(unique_dates) - 2)

    train_end = unique_dates.iloc[train_idx]
    val_end = unique_dates.iloc[val_idx]

    train = ordered[ordered["date"] <= train_end].copy()
    val = ordered[(ordered["date"] > train_end) & (ordered["date"] <= val_end)].copy()
    test = ordered[ordered["date"] > val_end].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            "Time split produced an empty partition. "
            f"Sizes: train={len(train)}, val={len(val)}, test={len(test)}"
        )

    return TimeSplit(train=train, val=val, test=test, train_end_date=train_end, val_end_date=val_end)
