import numpy as np
import pandas as pd

from src.simulation.config import SimulationConfig
from src.simulation.group_stage import simulate_group_stage
from src.simulation.match_simulator import MatchSimulator
from src.simulation.played_matches import (
    build_played_result_map,
    extract_wc26_group_matches,
    played_result_for_fixture,
)
from src.simulation.schemas import MatchFixture


def predictable_home_loss(home_team: str, away_team: str) -> dict[str, float]:
    return {
        "home_win_probability": 0.0,
        "draw_probability": 0.0,
        "away_win_probability": 1.0,
        "predicted_home_goals": 0.2,
        "predicted_away_goals": 2.0,
    }


def test_played_result_orients_to_fixture_order():
    played = pd.DataFrame(
        [
            {
                "date": "2026-06-12",
                "group": "A",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "home_goals": 2,
                "away_goals": 1,
                "source": "test",
            }
        ]
    )
    result_map = build_played_result_map(played)
    fixture = MatchFixture(
        home_team="South Africa",
        away_team="Mexico",
        stage="group",
        group="A",
    )

    result = played_result_for_fixture(fixture, result_map)

    assert result is not None
    assert result.home_team == "South Africa"
    assert result.away_team == "Mexico"
    assert result.home_goals == 1
    assert result.away_goals == 2
    assert result.winner == "Mexico"


def test_group_stage_uses_played_result_before_simulating():
    played = pd.DataFrame(
        [
            {
                "date": "2026-06-12",
                "group": "A",
                "home_team": "A1",
                "away_team": "A2",
                "home_goals": 4,
                "away_goals": 0,
                "source": "test",
            }
        ]
    )
    rng = np.random.default_rng(7)
    simulator = MatchSimulator(
        predict_match_fn=predictable_home_loss,
        rng=rng,
        config=SimulationConfig(num_groups=1, best_third_place_to_advance=0),
    )

    result = simulate_group_stage(
        group_name="A",
        teams=["A1", "A2", "A3", "A4"],
        match_simulator=simulator,
        rng=rng,
        played_results=build_played_result_map(played),
    )

    played_match = result.results[0]
    assert played_match.home_team == "A1"
    assert played_match.away_team == "A2"
    assert played_match.home_goals == 4
    assert played_match.away_goals == 0


def test_extract_wc26_group_matches_from_training_table():
    training = pd.DataFrame(
        [
            {
                "date": "2026-06-15",
                "home_team": "France",
                "away_team": "Senegal",
                "home_score": 1,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "tournament_type": "World Cup",
            },
            {
                "date": "2026-06-15",
                "home_team": "France",
                "away_team": "Brazil",
                "home_score": 2,
                "away_score": 2,
                "tournament": "FIFA World Cup",
                "tournament_type": "World Cup",
            },
        ]
    )

    out = extract_wc26_group_matches(
        training,
        groups={"I": ["France", "Senegal", "Norway", "Iraq"], "C": ["Brazil"]},
    )

    assert len(out) == 1
    assert out.iloc[0]["group"] == "I"
    assert out.iloc[0]["home_team"] == "France"
    assert out.iloc[0]["away_goals"] == 1
