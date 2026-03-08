"""Reporting helpers that convert simulation counts to DataFrames."""

from __future__ import annotations

import pandas as pd


def build_advancement_probabilities(
    progression_counts: dict[str, dict[str, int]],
    simulations: int,
) -> pd.DataFrame:
    """Convert progression counts to probabilities per team."""
    rows = []
    for team, counts in progression_counts.items():
        row = {"team": team}
        for key, value in counts.items():
            row[key] = float(value) / float(simulations)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "won_tournament" in df.columns:
        df = df.sort_values(["won_tournament", "reached_final"], ascending=[False, False])
    return df.reset_index(drop=True)


def build_champion_probabilities(
    advancement_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Extract champion probabilities table."""
    cols = ["team", "won_tournament"]
    existing = [col for col in cols if col in advancement_probabilities.columns]
    out = advancement_probabilities[existing].copy()
    out = out.rename(columns={"won_tournament": "champion_probability"})
    out = out.sort_values("champion_probability", ascending=False).reset_index(drop=True)
    return out


def build_group_winner_probabilities(
    group_winner_counts: dict[tuple[str, str], int],
    simulations: int,
) -> pd.DataFrame:
    """Convert group-winner counts to probabilities."""
    rows = []
    for (group_name, team), count in group_winner_counts.items():
        rows.append(
            {
                "group": group_name,
                "team": team,
                "group_winner_probability": float(count) / float(simulations),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group", "group_winner_probability"], ascending=[True, False]).reset_index(drop=True)
    return out
