import numpy as np

from src.simulation.bracket import build_seeded_knockout_fixtures, simulate_knockout_bracket
from src.simulation.match_simulator import MatchSimulator


def always_home_predictor(home_team: str, away_team: str) -> dict[str, float]:
    del home_team, away_team
    return {
        "home_win_probability": 1.0,
        "draw_probability": 0.0,
        "away_win_probability": 0.0,
        "predicted_home_goals": 2.0,
        "predicted_away_goals": 0.2,
    }


def test_knockout_advancement_pipeline():
    teams = [f"T{i}" for i in range(1, 9)]
    fixtures = build_seeded_knockout_fixtures(teams)

    simulator = MatchSimulator(always_home_predictor, rng=np.random.default_rng(5))
    knockout = simulate_knockout_bracket(fixtures, match_simulator=simulator, rng=np.random.default_rng(5))

    assert "quarterfinal" in knockout.participants_by_round
    assert "semifinal" in knockout.participants_by_round
    assert "final" in knockout.participants_by_round
    assert knockout.champion == "T1"
