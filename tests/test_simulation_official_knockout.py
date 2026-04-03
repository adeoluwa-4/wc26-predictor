import numpy as np
import pandas as pd

from src.simulation.bracket import (
    build_official_round_of_32_fixtures,
    resolve_third_place_slot_groups,
    simulate_official_knockout_bracket,
)
from src.simulation.config import SimulationConfig
from src.simulation.match_simulator import MatchSimulator
from src.simulation.tournament import run_single_tournament


def always_home_predictor(home_team: str, away_team: str) -> dict[str, float]:
    del home_team, away_team
    return {
        "home_win_probability": 1.0,
        "draw_probability": 0.0,
        "away_win_probability": 0.0,
        "predicted_home_goals": 2.0,
        "predicted_away_goals": 0.2,
    }


def _mock_group_finishers() -> dict[str, dict[str, str]]:
    out = {}
    for group in "ABCDEFGHIJKL":
        out[group] = {
            "winner": f"{group}_W",
            "runner_up": f"{group}_R",
            "third": f"{group}_T",
            "fourth": f"{group}_F",
        }
    return out


def _mock_selected_third() -> pd.DataFrame:
    groups = ["A", "C", "D", "E", "F", "H", "I", "J"]
    rows = []
    for rank, group in enumerate(groups, start=1):
        rows.append(
            {
                "group": group,
                "team": f"{group}_T",
                "points": 4,
                "goal_diff": 0,
                "goals_for": 3,
                "wins": 1,
                "third_place_rank": rank,
            }
        )
    return pd.DataFrame(rows)


def test_third_place_combination_resolution_is_deterministic_and_valid():
    selected_groups = ["A", "C", "D", "E", "F", "H", "I", "J"]
    mapping_one = resolve_third_place_slot_groups(selected_groups)
    mapping_two = resolve_third_place_slot_groups(selected_groups)

    assert mapping_one == mapping_two
    assert len(mapping_one) == 8
    assert len(set(mapping_one.values())) == 8
    assert set(mapping_one.values()) == set(selected_groups)


def test_round_of_32_slot_assignment_uses_fixed_match_numbers():
    group_finishers = _mock_group_finishers()
    selected_third = _mock_selected_third()
    fixtures, slot_groups = build_official_round_of_32_fixtures(group_finishers, selected_third)
    by_match = {int(f.match_number): f for f in fixtures}

    assert sorted(by_match.keys()) == list(range(73, 89))
    assert by_match[73].home_team == "A_R"
    assert by_match[73].away_team == "B_R"
    assert by_match[74].home_team == "E_W"
    assert by_match[74].away_team == f"{slot_groups[74]}_T"


def test_official_knockout_progression_paths_and_champion():
    group_finishers = _mock_group_finishers()
    selected_third = _mock_selected_third()
    fixtures, _ = build_official_round_of_32_fixtures(group_finishers, selected_third)

    cfg = SimulationConfig(enforce_neutral_order_invariance=False)
    simulator = MatchSimulator(always_home_predictor, rng=np.random.default_rng(7), config=cfg)
    out = simulate_official_knockout_bracket(fixtures, match_simulator=simulator)

    assert "round_of_32" in out.participants_by_round
    assert "round_of_16" in out.participants_by_round
    assert "quarterfinal" in out.participants_by_round
    assert "semifinal" in out.participants_by_round
    assert "final" in out.participants_by_round
    assert "third_place" in out.participants_by_round
    assert out.champion == "A_R"


def test_top2_plus_best8_third_advance_and_seed_is_deterministic():
    groups = {group: [f"{group}{i}" for i in range(1, 5)] for group in "ABCDEFGHIJKL"}
    cfg = SimulationConfig()

    run_one = run_single_tournament(groups, always_home_predictor, seed=1234, config=cfg)
    run_two = run_single_tournament(groups, always_home_predictor, seed=1234, config=cfg)

    qualified_count = sum(flags["qualified_from_group"] for flags in run_one.progression.values())
    third_adv_count = sum(flags["advanced_as_third_place"] for flags in run_one.progression.values())
    assert qualified_count == 32
    assert third_adv_count == 8

    assert run_one.knockout.champion == run_two.knockout.champion
    assert run_one.round_of_32_pairings == run_two.round_of_32_pairings
