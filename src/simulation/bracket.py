"""Knockout bracket construction and simulation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.simulation.knockout_config import (
    FINAL_PATH,
    QUARTERFINAL_PATHS,
    ROUND_OF_16_PATHS,
    ROUND_OF_32_SLOT_TOKENS,
    SEMIFINAL_PATHS,
    THIRD_PLACE_COMBINATION_LOOKUP,
    THIRD_PLACE_PATH,
)
from src.simulation.match_simulator import MatchSimulator
from src.simulation.schemas import KnockoutSimulationResult, MatchFixture, SimulatedMatchResult


ROUND_LABELS = {
    32: "round_of_32",
    16: "round_of_16",
    8: "quarterfinal",
    4: "semifinal",
    2: "final",
}


def stage_for_team_count(team_count: int) -> str:
    """Map active knockout team count to round label."""
    if team_count not in ROUND_LABELS:
        raise ValueError(f"Unsupported knockout team count: {team_count}")
    return ROUND_LABELS[team_count]


def build_seeded_knockout_fixtures(teams: list[str]) -> list[MatchFixture]:
    """Legacy generic seeding helper kept for debug/testing paths."""
    if len(teams) % 2 != 0:
        raise ValueError("Number of knockout teams must be even")

    stage = stage_for_team_count(len(teams))
    fixtures = []
    for i in range(len(teams) // 2):
        fixtures.append(MatchFixture(home_team=teams[i], away_team=teams[-1 - i], stage=stage, group=None))
    return fixtures


def _build_next_round_fixtures(winners: list[str]) -> list[MatchFixture]:
    stage = stage_for_team_count(len(winners))
    fixtures = []
    for i in range(0, len(winners), 2):
        fixtures.append(MatchFixture(home_team=winners[i], away_team=winners[i + 1], stage=stage, group=None))
    return fixtures


def simulate_knockout_bracket(
    initial_fixtures: list[MatchFixture],
    match_simulator: MatchSimulator,
    rng: np.random.Generator,
) -> KnockoutSimulationResult:
    """Legacy generic knockout simulator kept for compatibility tests."""
    del rng  # randomness is consumed by match_simulator

    if not initial_fixtures:
        raise ValueError("Need at least one knockout fixture")

    fixtures = initial_fixtures
    results_by_round: dict[str, list[SimulatedMatchResult]] = {}
    participants_by_round: dict[str, list[str]] = {}

    while fixtures:
        round_name = fixtures[0].stage
        participants: list[str] = []
        for fx in fixtures:
            participants.append(fx.home_team)
            participants.append(fx.away_team)
        participants_by_round[round_name] = participants

        round_results = []
        winners = []
        for fixture in fixtures:
            result = match_simulator.simulate_match(fixture, knockout=True)
            round_results.append(result)
            if result.winner is None:
                raise RuntimeError("Knockout match must produce a winner")
            winners.append(result.winner)

        results_by_round[round_name] = round_results

        if len(winners) == 1:
            champion = winners[0]
            final_participants = tuple(participants_by_round["final"])
            return KnockoutSimulationResult(
                results_by_round=results_by_round,
                participants_by_round=participants_by_round,
                champion=champion,
                finalists=(str(final_participants[0]), str(final_participants[1])),
            )

        fixtures = _build_next_round_fixtures(winners)

    raise RuntimeError("Knockout simulation ended unexpectedly")


def resolve_third_place_slot_groups(selected_groups: Iterable[str]) -> dict[int, str]:
    """Resolve best-third routing using deterministic 2026 combination lookup."""
    combo = tuple(sorted(str(group).upper() for group in selected_groups))
    if len(combo) != 8:
        raise ValueError(f"Expected 8 selected third-place groups, got {len(combo)}")
    if combo not in THIRD_PLACE_COMBINATION_LOOKUP:
        raise ValueError(f"No third-place routing found for group combination: {combo}")
    return THIRD_PLACE_COMBINATION_LOOKUP[combo].copy()


def _resolve_group_slot_token(group_finishers: dict[str, dict[str, str]], token: str) -> str:
    if len(token) != 2:
        raise ValueError(f"Invalid group slot token: {token}")

    group = token[0]
    position = token[1]
    if group not in group_finishers:
        raise ValueError(f"Group {group} not found in finishers")

    if position == "1":
        return group_finishers[group]["winner"]
    if position == "2":
        return group_finishers[group]["runner_up"]

    raise ValueError(f"Unsupported token position for {token}")


def build_official_round_of_32_fixtures(
    group_finishers: dict[str, dict[str, str]],
    selected_third_place: pd.DataFrame,
) -> tuple[list[MatchFixture], dict[int, str]]:
    """Build official match 73-88 fixtures with best-third routing."""
    if len(selected_third_place) != 8:
        raise ValueError(f"Expected 8 selected third-place teams, got {len(selected_third_place)}")

    third_df = selected_third_place.copy()
    third_df["group"] = third_df["group"].astype(str).str.upper()
    selected_groups = third_df["group"].tolist()
    slot_groups = resolve_third_place_slot_groups(selected_groups)

    third_by_group = dict(zip(third_df["group"], third_df["team"]))

    fixtures: list[MatchFixture] = []
    for match_number, (home_token, away_token) in ROUND_OF_32_SLOT_TOKENS.items():
        if home_token.startswith("TP"):
            home_team = third_by_group[slot_groups[int(home_token.replace("TP", ""))]]
        else:
            home_team = _resolve_group_slot_token(group_finishers, home_token)

        if away_token.startswith("TP"):
            away_team = third_by_group[slot_groups[int(away_token.replace("TP", ""))]]
        else:
            away_team = _resolve_group_slot_token(group_finishers, away_token)

        fixtures.append(
            MatchFixture(
                home_team=str(home_team),
                away_team=str(away_team),
                stage="round_of_32",
                group=None,
                match_number=int(match_number),
            )
        )

    validate_round_of_32_fixtures(fixtures)
    return sorted(fixtures, key=lambda fx: int(fx.match_number or 0)), slot_groups


def validate_round_of_32_fixtures(fixtures: list[MatchFixture]) -> None:
    """Validate the official R32 fixture card."""
    if len(fixtures) != 16:
        raise ValueError(f"Round of 32 must have 16 matches, got {len(fixtures)}")

    numbers = [int(fx.match_number or -1) for fx in fixtures]
    expected = list(range(73, 89))
    if sorted(numbers) != expected:
        raise ValueError(f"Round of 32 match numbers must be {expected}")

    teams = [team for fx in fixtures for team in (fx.home_team, fx.away_team)]
    if len(teams) != 32:
        raise ValueError("Round of 32 must include 32 teams")
    if len(set(teams)) != 32:
        raise ValueError("Round of 32 contains duplicate teams")


def _simulate_round(
    match_ids: list[int],
    fixtures_by_id: dict[int, MatchFixture],
    match_simulator: MatchSimulator,
) -> tuple[list[SimulatedMatchResult], dict[int, SimulatedMatchResult], list[str]]:
    round_results: list[SimulatedMatchResult] = []
    by_id: dict[int, SimulatedMatchResult] = {}
    participants: list[str] = []
    for match_id in match_ids:
        fixture = fixtures_by_id[match_id]
        participants.extend([fixture.home_team, fixture.away_team])
        result = match_simulator.simulate_match(fixture, knockout=True)
        if result.winner is None:
            raise RuntimeError(f"Knockout match {match_id} ended without winner")
        round_results.append(result)
        by_id[match_id] = result
    return round_results, by_id, participants


def _loser(result: SimulatedMatchResult) -> str:
    if result.winner == result.home_team:
        return result.away_team
    if result.winner == result.away_team:
        return result.home_team
    raise RuntimeError("Knockout result has no winner")


def validate_official_knockout_progression(
    round_of_32_fixtures: list[MatchFixture],
    results_by_match: dict[int, SimulatedMatchResult],
) -> None:
    """Validate knockout shape and winner-progression integrity."""
    validate_round_of_32_fixtures(round_of_32_fixtures)

    if len(results_by_match) < 32:
        raise ValueError("Expected at least 32 knockout match results (including third-place and final)")

    # Validate round construction from prior winners.
    for match_id, (left, right) in ROUND_OF_16_PATHS.items():
        result = results_by_match[match_id]
        if result.home_team != results_by_match[left].winner or result.away_team != results_by_match[right].winner:
            raise ValueError(f"Invalid Round of 16 progression for match {match_id}")

    for match_id, (left, right) in QUARTERFINAL_PATHS.items():
        result = results_by_match[match_id]
        if result.home_team != results_by_match[left].winner or result.away_team != results_by_match[right].winner:
            raise ValueError(f"Invalid quarterfinal progression for match {match_id}")

    for match_id, (left, right) in SEMIFINAL_PATHS.items():
        result = results_by_match[match_id]
        if result.home_team != results_by_match[left].winner or result.away_team != results_by_match[right].winner:
            raise ValueError(f"Invalid semifinal progression for match {match_id}")

    final = results_by_match[103]
    if final.home_team != results_by_match[FINAL_PATH[0]].winner or final.away_team != results_by_match[FINAL_PATH[1]].winner:
        raise ValueError("Invalid final progression")

    third_place = results_by_match[104]
    if third_place.home_team != _loser(results_by_match[THIRD_PLACE_PATH[0]]) or third_place.away_team != _loser(
        results_by_match[THIRD_PLACE_PATH[1]]
    ):
        raise ValueError("Invalid third-place progression")


def simulate_official_knockout_bracket(
    round_of_32_fixtures: list[MatchFixture],
    match_simulator: MatchSimulator,
) -> KnockoutSimulationResult:
    """Simulate official 2026 knockout path including third-place match."""
    validate_round_of_32_fixtures(round_of_32_fixtures)

    fixtures_by_id: dict[int, MatchFixture] = {int(fx.match_number): fx for fx in round_of_32_fixtures if fx.match_number is not None}
    results_by_match: dict[int, SimulatedMatchResult] = {}
    results_by_round: dict[str, list[SimulatedMatchResult]] = {}
    participants_by_round: dict[str, list[str]] = {}

    r32_ids = list(range(73, 89))
    r32_results, r32_by_id, r32_participants = _simulate_round(r32_ids, fixtures_by_id, match_simulator)
    results_by_match.update(r32_by_id)
    results_by_round["round_of_32"] = r32_results
    participants_by_round["round_of_32"] = r32_participants

    for match_id, (left, right) in ROUND_OF_16_PATHS.items():
        fixtures_by_id[match_id] = MatchFixture(
            home_team=results_by_match[left].winner or "",
            away_team=results_by_match[right].winner or "",
            stage="round_of_16",
            match_number=match_id,
        )
    r16_ids = sorted(ROUND_OF_16_PATHS.keys())
    r16_results, r16_by_id, r16_participants = _simulate_round(r16_ids, fixtures_by_id, match_simulator)
    results_by_match.update(r16_by_id)
    results_by_round["round_of_16"] = r16_results
    participants_by_round["round_of_16"] = r16_participants

    for match_id, (left, right) in QUARTERFINAL_PATHS.items():
        fixtures_by_id[match_id] = MatchFixture(
            home_team=results_by_match[left].winner or "",
            away_team=results_by_match[right].winner or "",
            stage="quarterfinal",
            match_number=match_id,
        )
    qf_ids = sorted(QUARTERFINAL_PATHS.keys())
    qf_results, qf_by_id, qf_participants = _simulate_round(qf_ids, fixtures_by_id, match_simulator)
    results_by_match.update(qf_by_id)
    results_by_round["quarterfinal"] = qf_results
    participants_by_round["quarterfinal"] = qf_participants

    for match_id, (left, right) in SEMIFINAL_PATHS.items():
        fixtures_by_id[match_id] = MatchFixture(
            home_team=results_by_match[left].winner or "",
            away_team=results_by_match[right].winner or "",
            stage="semifinal",
            match_number=match_id,
        )
    sf_ids = sorted(SEMIFINAL_PATHS.keys())
    sf_results, sf_by_id, sf_participants = _simulate_round(sf_ids, fixtures_by_id, match_simulator)
    results_by_match.update(sf_by_id)
    results_by_round["semifinal"] = sf_results
    participants_by_round["semifinal"] = sf_participants

    fixtures_by_id[103] = MatchFixture(
        home_team=results_by_match[FINAL_PATH[0]].winner or "",
        away_team=results_by_match[FINAL_PATH[1]].winner or "",
        stage="final",
        match_number=103,
    )
    final_results, final_by_id, final_participants = _simulate_round([103], fixtures_by_id, match_simulator)
    results_by_match.update(final_by_id)
    results_by_round["final"] = final_results
    participants_by_round["final"] = final_participants

    fixtures_by_id[104] = MatchFixture(
        home_team=_loser(results_by_match[THIRD_PLACE_PATH[0]]),
        away_team=_loser(results_by_match[THIRD_PLACE_PATH[1]]),
        stage="third_place",
        match_number=104,
    )
    third_results, third_by_id, third_participants = _simulate_round([104], fixtures_by_id, match_simulator)
    results_by_match.update(third_by_id)
    results_by_round["third_place"] = third_results
    participants_by_round["third_place"] = third_participants

    champion = results_by_match[103].winner
    if champion is None:
        raise RuntimeError("Final did not produce champion")
    finalists = (
        results_by_match[103].home_team,
        results_by_match[103].away_team,
    )
    validate_official_knockout_progression(round_of_32_fixtures, results_by_match)

    return KnockoutSimulationResult(
        results_by_round=results_by_round,
        participants_by_round=participants_by_round,
        champion=champion,
        finalists=finalists,
        third_place_winner=results_by_match[104].winner,
        results_by_match_number=results_by_match,
    )
