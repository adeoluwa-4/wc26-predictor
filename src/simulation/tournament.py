"""Single-run tournament orchestration for WC26 format."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.models.predict_interface import WC26Predictor
from src.simulation.bracket import (
    build_official_round_of_32_fixtures,
    build_seeded_knockout_fixtures,
    simulate_knockout_bracket,
    simulate_official_knockout_bracket,
)
from src.simulation.config import SimulationConfig
from src.simulation.group_stage import simulate_group_stage
from src.simulation.match_simulator import MatchSimulator
from src.simulation.schemas import MatchFixture, TournamentRunResult


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
    include_group_details: bool = True,
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
    third_rows = []
    group_winners = {}
    group_finishers: dict[str, dict[str, str]] = {}

    for group_name in ordered_groups:
        teams = groups[group_name]
        if len(teams) != cfg.group_size:
            raise ValueError(
                f"Group {group_name} must have exactly {cfg.group_size} teams, got {len(teams)}"
            )

        if include_group_details:
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
            group_finishers[group_name] = {
                "winner": str(standings.iloc[0]["team"]),
                "runner_up": str(standings.iloc[1]["team"]),
                "third": str(standings.iloc[2]["team"]),
                "fourth": str(standings.iloc[3]["team"]),
            }

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
        else:
            # Fast path: no pandas in inner loop; same tiebreak rules.
            stats = {
                team: {
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goal_diff": 0,
                    "points": 0,
                }
                for team in teams
            }

            # Round-robin: 6 fixtures for 4 teams.
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    home = teams[i]
                    away = teams[j]
                    fixture = MatchFixture(
                        home_team=home,
                        away_team=away,
                        stage="group",
                        group=group_name,
                    )
                    result = match_simulator.simulate_match(fixture, knockout=False)

                    h = stats[home]
                    a = stats[away]
                    h["played"] += 1
                    a["played"] += 1
                    h["goals_for"] += int(result.home_goals)
                    h["goals_against"] += int(result.away_goals)
                    a["goals_for"] += int(result.away_goals)
                    a["goals_against"] += int(result.home_goals)

                    if result.home_goals > result.away_goals:
                        h["wins"] += 1
                        a["losses"] += 1
                        h["points"] += int(cfg.points_for_win)
                    elif result.away_goals > result.home_goals:
                        a["wins"] += 1
                        h["losses"] += 1
                        a["points"] += int(cfg.points_for_win)
                    else:
                        h["draws"] += 1
                        a["draws"] += 1
                        h["points"] += int(cfg.points_for_draw)
                        a["points"] += int(cfg.points_for_draw)

                    h["goal_diff"] = h["goals_for"] - h["goals_against"]
                    a["goal_diff"] = a["goals_for"] - a["goals_against"]

            ranked = []
            for team in teams:
                row = stats[team]
                ranked.append(
                    (
                        -row["points"],
                        -row["goal_diff"],
                        -row["goals_for"],
                        -row["wins"],
                        float(rng.random()),
                        team,
                    )
                )
            ranked.sort()
            ordered_teams = [t for *_, t in ranked]

            group_winners[group_name] = ordered_teams[0]
            group_finishers[group_name] = {
                "winner": ordered_teams[0],
                "runner_up": ordered_teams[1],
                "third": ordered_teams[2],
                "fourth": ordered_teams[3],
            }

            third_team = ordered_teams[2]
            third = stats[third_team]
            third_rows.append(
                {
                    "group": group_name,
                    "team": third_team,
                    "points": int(third["points"]),
                    "goal_diff": int(third["goal_diff"]),
                    "goals_for": int(third["goals_for"]),
                    "wins": int(third["wins"]),
                }
            )

    third_df = pd.DataFrame(third_rows)
    ranked_third = rank_third_place_teams(third_df, rng=rng)
    selected_third_df = ranked_third.head(cfg.best_third_place_to_advance).copy().reset_index(drop=True)
    best_third = selected_third_df["team"].tolist()
    qualifiers = (
        [finishers["winner"] for finishers in group_finishers.values()]
        + [finishers["runner_up"] for finishers in group_finishers.values()]
        + best_third
    )
    qualifiers_set = set(qualifiers)
    best_third_set = set(best_third)

    use_official_bracket = (
        cfg.num_groups == 12 and cfg.group_size == 4 and cfg.best_third_place_to_advance == 8
    )

    if use_official_bracket:
        round_of_32_fixtures, third_slot_groups = build_official_round_of_32_fixtures(
            group_finishers=group_finishers,
            selected_third_place=selected_third_df,
        )
        knockout = simulate_official_knockout_bracket(
            round_of_32_fixtures=round_of_32_fixtures,
            match_simulator=match_simulator,
        )
        round_of_32_pairings = [
            {"match": str(int(fx.match_number or -1)), "home_team": fx.home_team, "away_team": fx.away_team}
            for fx in round_of_32_fixtures
        ]
    else:
        seeded_fixtures = build_seeded_knockout_fixtures(qualifiers)
        knockout = simulate_knockout_bracket(
            initial_fixtures=seeded_fixtures,
            match_simulator=match_simulator,
            rng=rng,
        )
        third_slot_groups = {}
        round_of_32_pairings = [
            {"match": str(idx + 1), "home_team": fx.home_team, "away_team": fx.away_team}
            for idx, fx in enumerate(seeded_fixtures)
        ]

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
        group_finishers=group_finishers,
        selected_third_place=selected_third_df,
        third_place_slot_groups=third_slot_groups,
        round_of_32_pairings=round_of_32_pairings,
    )
