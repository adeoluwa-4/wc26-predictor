from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.config import SimulationConfig
from src.simulation.monte_carlo import run_world_cup_simulation
from src.simulation.team_config import build_groups_from_team_config, load_team_config


def lightweight_predictor(home_team: str, away_team: str) -> dict[str, float]:
    del home_team, away_team
    return {
        "home_win_probability": 0.45,
        "draw_probability": 0.25,
        "away_win_probability": 0.30,
        "predicted_home_goals": 1.3,
        "predicted_away_goals": 1.1,
    }


def test_random_draw_from_wc26_team_config_respects_constraints():
    cfg = SimulationConfig()
    teams_df = load_team_config("data/config/wc26_teams.csv", config=cfg)
    groups = build_groups_from_team_config(teams_df, rng=np.random.default_rng(42), config=cfg)

    assert len(groups) == cfg.num_groups
    assert all(len(team_list) == cfg.group_size for team_list in groups.values())

    all_teams = [team for team_list in groups.values() for team in team_list]
    assert len(all_teams) == cfg.num_groups * cfg.group_size
    assert len(set(all_teams)) == cfg.num_groups * cfg.group_size

    team_meta = teams_df.set_index("team")[["confederation", "pot"]]
    for group_teams in groups.values():
        group_meta = team_meta.loc[group_teams]
        confed_counts = group_meta["confederation"].value_counts().to_dict()
        assert confed_counts.get("UEFA", 0) <= 2
        for confed, count in confed_counts.items():
            if confed != "UEFA":
                assert count <= 1
        assert set(group_meta["pot"].astype(int).tolist()) == {1, 2, 3, 4}


def test_preassigned_groups_are_honored():
    cfg = SimulationConfig(
        num_groups=2,
        group_size=2,
        best_third_place_to_advance=0,
        group_names=("A", "B"),
    )
    teams_df = pd.DataFrame(
        [
            {"team": "T1", "confederation": "UEFA", "pot": 1, "group": "A"},
            {"team": "T2", "confederation": "CAF", "pot": 2, "group": "A"},
            {"team": "T3", "confederation": "UEFA", "pot": 1, "group": "B"},
            {"team": "T4", "confederation": "AFC", "pot": 2, "group": "B"},
        ]
    )

    groups = build_groups_from_team_config(teams_df, rng=np.random.default_rng(7), config=cfg)
    assert groups == {"A": ["T1", "T2"], "B": ["T3", "T4"]}


def test_monte_carlo_uses_team_config_when_groups_not_provided(tmp_path: Path):
    cfg = SimulationConfig(
        num_groups=4,
        group_size=4,
        best_third_place_to_advance=0,
        group_names=("A", "B", "C", "D"),
        random_seed=17,
        teams_config_path=str(tmp_path / "teams.csv"),
    )
    rows = []
    confeds = ["UEFA", "CONMEBOL", "CAF", "AFC"]
    team_id = 1
    for pot in range(1, 5):
        for confed in confeds:
            rows.append({"team": f"T{team_id}", "confederation": confed, "pot": pot})
            team_id += 1
    pd.DataFrame(rows).to_csv(cfg.teams_config_path, index=False)

    result = run_world_cup_simulation(
        n_simulations=5,
        groups=None,
        config=cfg,
        predict_match_fn=lightweight_predictor,
    )
    assert len(result.advancement_probabilities) == 16
    assert set(result.raw["groups"].keys()) == {"A", "B", "C", "D"}
