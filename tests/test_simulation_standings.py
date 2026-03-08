import numpy as np
import pandas as pd

from src.simulation.standings import sort_group_table


def test_standings_sort_order_matches_tiebreak_rules():
    table = pd.DataFrame(
        [
            {"group": "A", "team": "T1", "played": 3, "wins": 1, "draws": 2, "losses": 0, "goals_for": 3, "goals_against": 2, "goal_diff": 1, "points": 5},
            {"group": "A", "team": "T2", "played": 3, "wins": 1, "draws": 2, "losses": 0, "goals_for": 3, "goals_against": 1, "goal_diff": 2, "points": 5},
            {"group": "A", "team": "T3", "played": 3, "wins": 1, "draws": 2, "losses": 0, "goals_for": 4, "goals_against": 2, "goal_diff": 2, "points": 5},
            {"group": "A", "team": "T4", "played": 3, "wins": 2, "draws": 0, "losses": 1, "goals_for": 4, "goals_against": 2, "goal_diff": 2, "points": 5},
        ]
    )

    out = sort_group_table(table, rng=np.random.default_rng(0))

    assert out["team"].tolist() == ["T4", "T3", "T2", "T1"]
    assert out["rank"].tolist() == [1, 2, 3, 4]
