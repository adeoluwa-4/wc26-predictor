import pandas as pd

from src.data.joins import join_strength_features


def test_asof_join_is_strictly_prior_not_same_day():
    matches = pd.DataFrame(
        {
            "match_id": ["M1"],
            "date": [pd.Timestamp("2020-01-10")],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [1],
            "away_score": [0],
        }
    )

    elo = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-09"), pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-09")],
            "team": ["A", "A", "B"],
            "elo_rating": [1490, 1600, 1400],
        }
    )

    fifa = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-09"), pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-09")],
            "team": ["A", "A", "B"],
            "fifa_rank": [12, 8, 20],
            "fifa_points": [1500.0, 1700.0, 1300.0],
        }
    )

    out, report = join_strength_features(matches, elo, fifa)

    # Must use 2020-01-09 snapshot for team A, not 2020-01-10.
    assert out.loc[0, "home_elo"] == 1490
    assert out.loc[0, "home_fifa_rank"] == 12
    assert report["home_elo_coverage"] == 1.0
