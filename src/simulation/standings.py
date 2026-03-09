"""Group standings table logic and sorting rules."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.simulation.config import SimulationConfig
from src.simulation.schemas import SimulatedMatchResult


def initialize_group_table(teams: Iterable[str], group_name: str) -> pd.DataFrame:
    """Create a fresh group standings table."""
    idx = list(teams)
    df = pd.DataFrame(
        {
            "group": group_name,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "points": 0,
        },
        index=pd.Index(idx, name="team"),
    )
    return df


def apply_match_result(
    table: pd.DataFrame,
    result: SimulatedMatchResult,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Apply one result to a group table and return updated table."""
    cfg = config or SimulationConfig()
    if table.index.name != "team":
        raise ValueError("Group table must be indexed by team")

    home = result.home_team
    away = result.away_team
    if home not in table.index or away not in table.index:
        raise ValueError("Both teams must exist in the group table index")

    table.at[home, "played"] += 1
    table.at[away, "played"] += 1

    table.at[home, "goals_for"] += result.home_goals
    table.at[home, "goals_against"] += result.away_goals
    table.at[away, "goals_for"] += result.away_goals
    table.at[away, "goals_against"] += result.home_goals

    if result.home_goals > result.away_goals:
        table.at[home, "wins"] += 1
        table.at[away, "losses"] += 1
        table.at[home, "points"] += cfg.points_for_win
    elif result.away_goals > result.home_goals:
        table.at[away, "wins"] += 1
        table.at[home, "losses"] += 1
        table.at[away, "points"] += cfg.points_for_win
    else:
        table.at[home, "draws"] += 1
        table.at[away, "draws"] += 1
        table.at[home, "points"] += cfg.points_for_draw
        table.at[away, "points"] += cfg.points_for_draw

    table.at[home, "goal_diff"] = table.at[home, "goals_for"] - table.at[home, "goals_against"]
    table.at[away, "goal_diff"] = table.at[away, "goals_for"] - table.at[away, "goals_against"]
    return table


def sort_group_table(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sort using v1 tiebreak order with random final fallback.

    Order:
    1. points
    2. goal difference
    3. goals for
    4. wins
    5. random tie break
    """
    out = table.copy()
    if "team" not in out.columns:
        out = out.reset_index()
    else:
        out = out.reset_index(drop=True)
    out["_random_tiebreak"] = rng.random(len(out))

    out = out.sort_values(
        by=["points", "goal_diff", "goals_for", "wins", "_random_tiebreak"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    out["rank"] = out.index + 1
    out = out.drop(columns=["_random_tiebreak"])
    return out
