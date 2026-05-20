import numpy as np

from src.simulation.match_simulator import MatchSimulator


def test_neutral_symmetric_prediction_removes_order_bias():
    def fake_predict(home: str, away: str) -> dict[str, float]:
        if (home, away) == ("A", "B"):
            return {
                "home_win_probability": 0.7,
                "draw_probability": 0.2,
                "away_win_probability": 0.1,
                "predicted_home_goals": 2.0,
                "predicted_away_goals": 0.7,
            }
        if (home, away) == ("B", "A"):
            return {
                "home_win_probability": 0.4,
                "draw_probability": 0.3,
                "away_win_probability": 0.3,
                "predicted_home_goals": 1.4,
                "predicted_away_goals": 1.1,
            }
        raise AssertionError("Unexpected pair")

    sim = MatchSimulator(predict_match_fn=fake_predict, rng=np.random.default_rng(0))
    pred_ab = sim._predict_neutral_symmetric("A", "B")
    pred_ba = sim._predict_neutral_symmetric("B", "A")

    assert round(pred_ab["home_win_probability"], 8) == round(pred_ba["away_win_probability"], 8)
    assert round(pred_ab["away_win_probability"], 8) == round(pred_ba["home_win_probability"], 8)
    assert round(pred_ab["draw_probability"], 8) == round(pred_ba["draw_probability"], 8)
    assert round(pred_ab["predicted_home_goals"], 8) == round(pred_ba["predicted_away_goals"], 8)
    assert round(pred_ab["predicted_away_goals"], 8) == round(pred_ba["predicted_home_goals"], 8)
