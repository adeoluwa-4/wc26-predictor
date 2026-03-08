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


@dataclass
class TournamentRunResult:
    """Outputs from one complete tournament simulation."""

    group_results: dict[str, GroupSimulationResult]
    third_place_ranking: pd.DataFrame
    knockout: KnockoutSimulationResult
    group_winners: dict[str, str]
    progression: dict[str, dict[str, bool]]


@dataclass
class MonteCarloResult:
    """Aggregated outputs from many tournament simulations."""

    advancement_probabilities: pd.DataFrame
    champion_probabilities: pd.DataFrame
    group_winner_probabilities: pd.DataFrame
    raw: dict[str, Any]
