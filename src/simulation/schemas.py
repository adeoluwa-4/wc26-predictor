"""Dataclasses used by simulation modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MatchFixture:
    """Single fixture definition."""

    home_team: str
    away_team: str
    stage: str
    group: str | None = None
    match_number: int | None = None


@dataclass(frozen=True)
class SimulatedMatchResult:
    """Single simulated match output."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    stage: str
    group: str | None
    winner: str | None
    is_draw: bool
    decided_by_penalties: bool


@dataclass
class GroupSimulationResult:
    """Outputs from one group-stage simulation."""

    group_name: str
    fixtures: list[MatchFixture]
    results: list[SimulatedMatchResult]
    standings: pd.DataFrame


@dataclass
class KnockoutSimulationResult:
    """Outputs from knockout bracket simulation."""

    results_by_round: dict[str, list[SimulatedMatchResult]]
    participants_by_round: dict[str, list[str]]
    champion: str
    finalists: tuple[str, str]
    third_place_winner: str | None = None
    results_by_match_number: dict[int, SimulatedMatchResult] | None = None


@dataclass
class TournamentRunResult:
    """Outputs from one complete tournament simulation."""

    group_results: dict[str, GroupSimulationResult]
    third_place_ranking: pd.DataFrame
    knockout: KnockoutSimulationResult
    group_winners: dict[str, str]
    progression: dict[str, dict[str, bool]]
    group_finishers: dict[str, dict[str, str]] | None = None
    selected_third_place: pd.DataFrame | None = None
    third_place_slot_groups: dict[int, str] | None = None
    round_of_32_pairings: list[dict[str, str]] | None = None


@dataclass
class MonteCarloResult:
    """Aggregated outputs from many tournament simulations."""

    advancement_probabilities: pd.DataFrame
    champion_probabilities: pd.DataFrame
    group_winner_probabilities: pd.DataFrame
    raw: dict[str, Any]
