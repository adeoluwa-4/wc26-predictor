"""Group-stage fixture generation and simulation."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.simulation.config import SimulationConfig
from src.simulation.match_simulator import MatchSimulator
from src.simulation.schemas import GroupSimulationResult, MatchFixture
from src.simulation.standings import apply_match_result, initialize_group_table, sort_group_table


def generate_group_fixtures(group_name: str, teams: list[str]) -> list[MatchFixture]:
    """Generate round-robin fixtures for one group of 4 teams."""
    fixtures: list[MatchFixture] = []
    for home, away in combinations(teams, 2):
        fixtures.append(MatchFixture(home_team=home, away_team=away, stage="group", group=group_name))
    return fixtures


def simulate_group_stage(
    group_name: str,
    teams: list[str],
    match_simulator: MatchSimulator,
    rng: np.random.Generator,
    config: SimulationConfig | None = None,
) -> GroupSimulationResult:
    """Simulate every fixture in one group and return final standings."""
    cfg = config or SimulationConfig()
    table = initialize_group_table(teams, group_name=group_name)
    fixtures = generate_group_fixtures(group_name=group_name, teams=teams)

    results = []
    for fixture in fixtures:
        result = match_simulator.simulate_match(fixture, knockout=False)
        table = apply_match_result(table, result, config=cfg)
        results.append(result)

    standings = sort_group_table(table, rng=rng)

    return GroupSimulationResult(
        group_name=group_name,
        fixtures=fixtures,
        results=results,
        standings=standings,
    )
