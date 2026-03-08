from src.simulation.config import SimulationConfig
from src.simulation.monte_carlo import run_world_cup_simulation


def lightweight_predictor(home_team: str, away_team: str) -> dict[str, float]:
    if home_team < away_team:
        home_win = 0.65
        away_win = 0.20
    else:
        home_win = 0.35
        away_win = 0.50

    return {
        "home_win_probability": home_win,
        "draw_probability": 0.15,
        "away_win_probability": away_win,
        "predicted_home_goals": 1.4,
        "predicted_away_goals": 1.1,
    }


def test_monte_carlo_aggregation_outputs():
    groups = {
        "A": ["A1", "A2", "A3", "A4"],
        "B": ["B1", "B2", "B3", "B4"],
        "C": ["C1", "C2", "C3", "C4"],
        "D": ["D1", "D2", "D3", "D4"],
    }

    cfg = SimulationConfig(
        num_groups=4,
        group_size=4,
        best_third_place_to_advance=0,
        group_names=("A", "B", "C", "D"),
        random_seed=11,
    )

    result = run_world_cup_simulation(
        n_simulations=20,
        groups=groups,
        config=cfg,
        predict_match_fn=lightweight_predictor,
    )

    adv = result.advancement_probabilities
    champ = result.champion_probabilities
    group_winners = result.group_winner_probabilities

    assert len(adv) == 16
    assert "won_tournament" in adv.columns
    assert abs(champ["champion_probability"].sum() - 1.0) < 1e-9
    assert set(group_winners["group"].unique()) == {"A", "B", "C", "D"}
