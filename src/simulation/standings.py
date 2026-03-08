"""Group standings table logic and sorting rules."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.simulation.config import SimulationConfig
from src.simulation.schemas import SimulatedMatchResult


def initialize_group_table(teams: Iterable[str], group_name: str) -> pd.DataFrame:
    """Create a fresh group standings table."""
    rows = []
    for team in teams:
        rows.append(
            {
                "group": group_name,
                "team": team,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_diff": 0,
                "points": 0,
            }
        )
    return pd.DataFrame(rows)


def apply_match_result(
    table: pd.DataFrame,
    result: SimulatedMatchResult,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Apply one result to a group table and return updated table."""
    cfg = config or SimulationConfig()
    out = table.copy()

    home_idx = out.index[out["team"] == result.home_team]
    away_idx = out.index[out["team"] == result.away_team]

    if len(home_idx) != 1 or len(away_idx) != 1:
        raise ValueError("Both teams must exist exactly once in the group table")

    h = int(home_idx[0])
    a = int(away_idx[0])

    out.at[h, "played"] += 1
    out.at[a, "played"] += 1

    out.at[h, "goals_for"] += result.home_goals
    out.at[h, "goals_against"] += result.away_goals
    out.at[a, "goals_for"] += result.away_goals
    out.at[a, "goals_against"] += result.home_goals

    if result.home_goals > result.away_goals:
        out.at[h, "wins"] += 1
        out.at[a, "losses"] += 1
        out.at[h, "points"] += cfg.points_for_win
    elif result.away_goals > result.home_goals:
        out.at[a, "wins"] += 1
        out.at[h, "losses"] += 1
        out.at[a, "points"] += cfg.points_for_win
    else:
        out.at[h, "draws"] += 1
        out.at[a, "draws"] += 1
        out.at[h, "points"] += cfg.points_for_draw
        out.at[a, "points"] += cfg.points_for_draw

    out["goal_diff"] = out["goals_for"] - out["goals_against"]
    return out


def sort_group_table(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sort using v1 tiebreak order with random final fallback.

    Order:
    1. points
    2. goal difference
    3. goals for
    4. wins
    5. random tie break
    """
    out = table.copy().reset_index(drop=True)
    out["_random_tiebreak"] = rng.random(len(out))

    out = out.sort_values(
        by=["points", "goal_diff", "goals_for", "wins", "_random_tiebreak"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    out["rank"] = out.index + 1
    out = out.drop(columns=["_random_tiebreak"])
    return out
