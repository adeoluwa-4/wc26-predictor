import pandas as pd

from src.data.rolling_features import add_rolling_features


def test_rolling_features_exclude_current_and_same_date_matches():
    matches = pd.DataFrame(
        {
            "match_id": ["M1", "M2", "M3"],
            "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "home_score": [1, 0, 0],
            "away_score": [0, 0, 1],
        }
    )

    out = add_rolling_features(matches, windows=(5,))

    # First match has no prior data.
    assert out.loc[out["match_id"] == "M1", "home_points_last_5"].iloc[0] == 0

    # Both same-day matches should only see prior date (M1) and not each other.
    assert out.loc[out["match_id"] == "M2", "home_points_last_5"].iloc[0] == 3
    assert out.loc[out["match_id"] == "M3", "home_points_last_5"].iloc[0] == 3
