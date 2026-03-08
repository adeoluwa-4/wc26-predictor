import numpy as np
import pandas as pd

from src.simulation.tournament import rank_third_place_teams


def test_third_place_ranking_order():
    third_rows = pd.DataFrame(
        [
            {"group": "A", "team": "A3", "points": 6, "goal_diff": 1, "goals_for": 5, "wins": 2},
            {"group": "B", "team": "B3", "points": 6, "goal_diff": 1, "goals_for": 4, "wins": 2},
            {"group": "C", "team": "C3", "points": 5, "goal_diff": 3, "goals_for": 6, "wins": 1},
            {"group": "D", "team": "D3", "points": 6, "goal_diff": 0, "goals_for": 9, "wins": 2},
            {"group": "E", "team": "E3", "points": 4, "goal_diff": 0, "goals_for": 3, "wins": 1},
        ]
    )

    ranked = rank_third_place_teams(third_rows, rng=np.random.default_rng(1))

    assert ranked["team"].tolist()[:4] == ["A3", "B3", "D3", "C3"]
    assert ranked.iloc[0]["third_place_rank"] == 1
