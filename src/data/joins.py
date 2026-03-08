"""Date-safe feature joins for Elo and FIFA tables."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _asof_join_team_features(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    team_col: str,
    feature_cols: Iterable[str],
    prefix: str,
) -> pd.DataFrame:
    """Join latest prior team features using strict backward as-of merge."""
    feature_cols = list(feature_cols)

    left = matches[["match_id", "date", team_col]].rename(columns={team_col: "team"}).copy()
    left = left.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

    right = features[["team", "date", *feature_cols]].copy()
    right = right.sort_values(["team", "date"]).reset_index(drop=True)

    merged_parts: list[pd.DataFrame] = []
    right_grouped = {team: grp.drop(columns=["team"]).sort_values("date") for team, grp in right.groupby("team")}

    for team, left_team in left.groupby("team", sort=False):
        left_team = left_team.sort_values(["date", "match_id"]).copy()
        right_team = right_grouped.get(team)

        if right_team is None or right_team.empty:
            continue

        merged_team = pd.merge_asof(
            left_team[["match_id", "date"]],
            right_team,
            on="date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_parts.append(merged_team)

    if merged_parts:
        merged = pd.concat(merged_parts, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=["match_id", "date", *feature_cols])

    rename_map = {col: f"{prefix}_{col}" for col in feature_cols}
    merged = merged.rename(columns=rename_map)
    merged = merged.drop(columns=["date"])

    return matches.merge(merged, on="match_id", how="left")


def join_strength_features(
    matches: pd.DataFrame,
    elo: pd.DataFrame,
    fifa: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach strict prior-date Elo and FIFA features for home/away teams."""
    df = matches.copy()

    df = _asof_join_team_features(df, elo, "home_team", ["elo_rating"], "home")
    df = _asof_join_team_features(df, elo, "away_team", ["elo_rating"], "away")
    df = df.rename(columns={"home_elo_rating": "home_elo", "away_elo_rating": "away_elo"})

    df = _asof_join_team_features(df, fifa, "home_team", ["fifa_rank", "fifa_points"], "home")
    df = _asof_join_team_features(df, fifa, "away_team", ["fifa_rank", "fifa_points"], "away")

    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["fifa_rank_diff"] = df["home_fifa_rank"] - df["away_fifa_rank"]
    df["fifa_points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]

    join_report = {
        "home_elo_coverage": float(df["home_elo"].notna().mean()),
        "away_elo_coverage": float(df["away_elo"].notna().mean()),
        "home_fifa_rank_coverage": float(df["home_fifa_rank"].notna().mean()),
        "away_fifa_rank_coverage": float(df["away_fifa_rank"].notna().mean()),
        "home_fifa_points_coverage": float(df["home_fifa_points"].notna().mean()),
        "away_fifa_points_coverage": float(df["away_fifa_points"].notna().mean()),
    }

    return df, join_report
