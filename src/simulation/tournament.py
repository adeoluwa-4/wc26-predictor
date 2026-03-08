"""Single-run tournament orchestration for WC26 format."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.models.predict_interface import WC26Predictor
from src.simulation.bracket import build_seeded_knockout_fixtures, simulate_knockout_bracket
from src.simulation.config import SimulationConfig
from src.simulation.group_stage import simulate_group_stage
from src.simulation.match_simulator import MatchSimulator
from src.simulation.schemas import TournamentRunResult


def build_default_groups(
    predictor: WC26Predictor,
    config: SimulationConfig | None = None,
) -> dict[str, list[str]]:
    """Build default groups from top Elo teams, preferring recently active teams."""
    cfg = config or SimulationConfig()
    cfg.validate()

    profiles = predictor.team_profiles.reset_index().copy()
    if "elo" not in profiles.columns:
        raise ValueError("Predictor team profiles are missing elo column")

    total_teams_needed = cfg.num_groups * cfg.group_size
    pool = profiles
    if cfg.min_active_date and "date" in profiles.columns:
        cutoff = pd.to_datetime(cfg.min_active_date, errors="coerce")
        if pd.notna(cutoff):
            active_pool = profiles[pd.to_datetime(profiles["date"], errors="coerce") >= cutoff].copy()
            if len(active_pool) >= total_teams_needed:
                pool = active_pool

    ranked = pool.sort_values("elo", ascending=False).head(total_teams_needed)
    teams = ranked["team"].tolist()

    if len(teams) < total_teams_needed:
        raise ValueError(
            f"Not enough teams in profiles for default groups. Need {total_teams_needed}, got {len(teams)}"
        )

    groups: dict[str, list[str]] = {cfg.group_names[i]: [] for i in range(cfg.num_groups)}

    # Snake-style seeding keeps top teams spread across groups.
    for row in range(cfg.group_size):
        slice_start = row * cfg.num_groups
        slice_end = (row + 1) * cfg.num_groups
        row_teams = teams[slice_start:slice_end]
        if row % 2 == 1:
            row_teams = list(reversed(row_teams))

        for group_idx, team in enumerate(row_teams):
            group_name = cfg.group_names[group_idx]
            groups[group_name].append(team)

    return groups


def rank_third_place_teams(third_rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rank third-place teams across groups.

    Order:
    1. points
    2. goal difference
    3. goals for
    4. wins
    5. random tie break
    """
    ranked = third_rows.copy().reset_index(drop=True)
    ranked["_random_tiebreak"] = rng.random(len(ranked))
    ranked = ranked.sort_values(
        by=["points", "goal_diff", "goals_for", "wins", "_random_tiebreak"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    ranked["third_place_rank"] = ranked.index + 1
    ranked = ranked.drop(columns=["_random_tiebreak"])
    return ranked


def run_single_tournament(
    groups: dict[str, list[str]],
    predict_match_fn: Callable[[str, str], dict[str, float]],
    seed: int,
    config: SimulationConfig | None = None,
) -> TournamentRunResult:
    """Run one full tournament simulation from groups to champion."""
    cfg = config or SimulationConfig()
    cfg.validate()

    rng = np.random.default_rng(seed)
    match_simulator = MatchSimulator(predict_match_fn=predict_match_fn, rng=rng, config=cfg)

    ordered_groups = [name for name in cfg.group_names[: cfg.num_groups] if name in groups]
    if len(ordered_groups) != cfg.num_groups:
        raise ValueError("Provided groups must include exactly the configured group names")

    group_results = {}
    seeded_top_two: list[str] = []
    third_rows = []
    group_winners = {}

    for group_name in ordered_groups:
        teams = groups[group_name]
        if len(teams) != cfg.group_size:
            raise ValueError(
                f"Group {group_name} must have exactly {cfg.group_size} teams, got {len(teams)}"
            )

        result = simulate_group_stage(
            group_name=group_name,
            teams=teams,
            match_simulator=match_simulator,
            rng=rng,
            config=cfg,
        )
        group_results[group_name] = result

        standings = result.standings
        group_winners[group_name] = str(standings.iloc[0]["team"])

        seeded_top_two.extend([str(standings.iloc[0]["team"]), str(standings.iloc[1]["team"])])

        third = standings.iloc[2]
        third_rows.append(
            {
                "group": group_name,
                "team": str(third["team"]),
                "points": int(third["points"]),
                "goal_diff": int(third["goal_diff"]),
                "goals_for": int(third["goals_for"]),
                "wins": int(third["wins"]),
            }
        )

    third_df = pd.DataFrame(third_rows)
    ranked_third = rank_third_place_teams(third_df, rng=rng)
    best_third = ranked_third.head(cfg.best_third_place_to_advance)["team"].tolist()

    qualifiers = seeded_top_two + best_third
    qualifiers_set = set(qualifiers)
    best_third_set = set(best_third)
    initial_fixtures = build_seeded_knockout_fixtures(qualifiers)
    knockout = simulate_knockout_bracket(initial_fixtures=initial_fixtures, match_simulator=match_simulator, rng=rng)

    stage_fields = [
        "qualified_from_group",
        "advanced_as_third_place",
        "reached_round_of_32",
        "reached_round_of_16",
        "reached_quarterfinal",
        "reached_semifinal",
        "reached_final",
        "won_tournament",
    ]

    all_teams = sorted({team for teams in groups.values() for team in teams})
    progression = {team: {field: False for field in stage_fields} for team in all_teams}

    for team in qualifiers_set:
        progression[team]["qualified_from_group"] = True
    for team in best_third_set:
        progression[team]["advanced_as_third_place"] = True

    round_to_field = {
        "round_of_32": "reached_round_of_32",
        "round_of_16": "reached_round_of_16",
        "quarterfinal": "reached_quarterfinal",
        "semifinal": "reached_semifinal",
        "final": "reached_final",
    }

    for round_name, participants in knockout.participants_by_round.items():
        if round_name not in round_to_field:
            continue
        field = round_to_field[round_name]
        for team in participants:
            progression[team][field] = True

    progression[knockout.champion]["won_tournament"] = True

    return TournamentRunResult(
        group_results=group_results,
        third_place_ranking=ranked_third,
        knockout=knockout,
        group_winners=group_winners,
        progression=progression,
    )
