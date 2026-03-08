"""Knockout bracket construction and simulation."""

from __future__ import annotations

import numpy as np

from src.simulation.match_simulator import MatchSimulator
from src.simulation.schemas import KnockoutSimulationResult, MatchFixture


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
    """Pair seeded teams as top-vs-bottom for the first knockout round."""
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
    """Simulate full knockout bracket until champion."""
    del rng  # randomness is consumed by match_simulator

    if not initial_fixtures:
        raise ValueError("Need at least one knockout fixture")

    fixtures = initial_fixtures
    results_by_round: dict[str, list] = {}
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
