import pandas as pd

from src.data.standardize import normalize_key, standardize_datasets


def test_normalize_key_basic():
    assert normalize_key("  U.S.A.  ") == "u.s.a."


def test_standardize_applies_former_and_overrides():
    results = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "home_team": ["Upper Volta"],
            "away_team": ["USA"],
            "home_score": [1],
            "away_score": [0],
            "tournament": ["Friendly"],
            "city": ["X"],
            "country": ["Burkina Faso"],
            "neutral": [False],
        }
    )

    shootouts = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "home_team": ["Upper Volta"],
            "away_team": ["USA"],
            "winner": ["USA"],
        }
    )

    former_names = pd.DataFrame(
        {
            "current": ["Burkina Faso"],
            "former": ["Upper Volta"],
            "start_date": [pd.Timestamp("1960-01-01")],
            "end_date": [pd.Timestamp("1984-08-04")],
        }
    )

    elo = pd.DataFrame(
        {
            "date": [pd.Timestamp("2019-12-31")],
            "team": ["USA"],
            "elo_rating": [1700],
        }
    )

    fifa = pd.DataFrame(
        {
            "date": [pd.Timestamp("2019-12-31")],
            "team": ["USA"],
            "fifa_rank": [10],
            "fifa_points": [1600.0],
        }
    )

    overrides = pd.DataFrame(
        {
            "source_name": ["USA"],
            "canonical_name": ["United States"],
        }
    )

    out = standardize_datasets(results, shootouts, former_names, elo, fifa, overrides=overrides)

    assert out.results.loc[0, "home_team"] == "Burkina Faso"
    assert out.results.loc[0, "away_team"] == "United States"
    assert out.shootouts.loc[0, "winner"] == "United States"
