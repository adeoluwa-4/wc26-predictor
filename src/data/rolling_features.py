"""Rolling team-form features computed from strictly prior matches."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import pandas as pd

BASE_METRICS = ("points", "goals_for", "goals_against", "goal_diff", "wins", "draws", "losses")


def build_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert match-level table into team-level long history (2 rows per match)."""
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": 1,
            "goals_for": matches["home_score"],
            "goals_against": matches["away_score"],
        }
    )

    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": 0,
            "goals_for": matches["away_score"],
            "goals_against": matches["home_score"],
        }
    )

    history = pd.concat([home, away], ignore_index=True)
    history["goal_diff"] = history["goals_for"] - history["goals_against"]
    history["wins"] = (history["goal_diff"] > 0).astype(int)
    history["draws"] = (history["goal_diff"] == 0).astype(int)
    history["losses"] = (history["goal_diff"] < 0).astype(int)
    history["points"] = history["wins"] * 3 + history["draws"]

    return history.sort_values(["team", "date", "match_id", "is_home"]).reset_index(drop=True)


def _window_sums(prior_records: list[dict[str, int]], window: int) -> dict[str, float]:
    """Compute sum metrics over last N prior records."""
    if not prior_records:
        return {metric: 0.0 for metric in BASE_METRICS} | {"matches": 0.0}

    subset = prior_records[-window:]
    sums = {metric: float(sum(rec[metric] for rec in subset)) for metric in BASE_METRICS}
    sums["matches"] = float(len(subset))
    return sums


def compute_team_rolling_features(team_history: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """Compute leakage-safe rolling features by team and date.

    All matches on the same date receive feature values derived from earlier dates only.
    """
    windows = tuple(sorted(set(int(w) for w in windows)))
    max_window = max(windows)

    df = team_history.sort_values(["team", "date", "match_id", "is_home"]).copy()

    feature_cols: list[str] = []
    for window in windows:
        for metric in BASE_METRICS:
            feature_cols.append(f"{metric}_last_{window}")
        feature_cols.append(f"matches_played_last_{window}")
        feature_cols.append(f"points_per_match_last_{window}")

    for col in feature_cols:
        df[col] = 0.0

    for team, team_df in df.groupby("team", sort=False):
        team_indices = team_df.index
        prior_queue: deque[dict[str, int]] = deque(maxlen=max_window)

        # Process one date bucket at a time to prevent same-date leakage.
        for _, date_bucket in team_df.groupby("date", sort=True):
            prior_list = list(prior_queue)

            cached_window_sums: dict[int, dict[str, float]] = {}
            for window in windows:
                cached_window_sums[window] = _window_sums(prior_list, window)

            for idx in date_bucket.index:
                for window in windows:
                    sums = cached_window_sums[window]
                    for metric in BASE_METRICS:
                        df.at[idx, f"{metric}_last_{window}"] = sums[metric]

                    matches_played = sums["matches"]
                    df.at[idx, f"matches_played_last_{window}"] = matches_played
                    ppm = (sums["points"] / matches_played) if matches_played > 0 else 0.0
                    df.at[idx, f"points_per_match_last_{window}"] = ppm

            for idx in date_bucket.index:
                prior_queue.append(
                    {
                        "points": int(df.at[idx, "points"]),
                        "goals_for": int(df.at[idx, "goals_for"]),
                        "goals_against": int(df.at[idx, "goals_against"]),
                        "goal_diff": int(df.at[idx, "goal_diff"]),
                        "wins": int(df.at[idx, "wins"]),
                        "draws": int(df.at[idx, "draws"]),
                        "losses": int(df.at[idx, "losses"]),
                    }
                )

    # Keep original order as in input by sorting on match_id then is_home for deterministic merges.
    df = df.sort_values(["match_id", "is_home"]).reset_index(drop=True)
    return df


def add_rolling_features(matches: pd.DataFrame, windows: Sequence[int] = (5, 10)) -> pd.DataFrame:
    """Add home/away rolling form features to match-level table."""
    team_history = build_team_history(matches)
    history_with_roll = compute_team_rolling_features(team_history, windows=windows)

    feature_cols = [
        col
        for col in history_with_roll.columns
        if any(col.endswith(f"_last_{w}") for w in windows)
        or any(col == f"points_per_match_last_{w}" for w in windows)
    ]

    home_features = (
        history_with_roll[history_with_roll["is_home"] == 1][["match_id", *feature_cols]]
        .rename(columns={col: f"home_{col}" for col in feature_cols})
        .reset_index(drop=True)
    )

    away_features = (
        history_with_roll[history_with_roll["is_home"] == 0][["match_id", *feature_cols]]
        .rename(columns={col: f"away_{col}" for col in feature_cols})
        .reset_index(drop=True)
    )

    merged = matches.merge(home_features, on="match_id", how="left")
    merged = merged.merge(away_features, on="match_id", how="left")

    # Drop helper columns not requested in final feature set.
    drop_cols = []
    for window in windows:
        drop_cols.extend(
            [
                f"home_matches_played_last_{window}",
                f"away_matches_played_last_{window}",
            ]
        )
    merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])

    return merged
