import pandas as pd

from src.models.features import get_feature_columns, make_time_split


def test_make_time_split_is_chronological():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
            ]),
            "home_team": ["A"] * 6,
            "away_team": ["B"] * 6,
            "tournament_type": ["Friendly"] * 6,
            "confederation_home": ["UEFA"] * 6,
            "confederation_away": ["UEFA"] * 6,
            "neutral": [True] * 6,
            "is_friendly": [True] * 6,
            "is_qualifier": [False] * 6,
            "is_continental_competition": [False] * 6,
            "is_world_cup": [False] * 6,
            "is_host_home_country": [False] * 6,
            "is_host_away_country": [False] * 6,
            "same_confederation": [True] * 6,
            "tournament_importance_score": [0.3] * 6,
            "home_elo": [1500] * 6,
            "away_elo": [1500] * 6,
            "elo_diff": [0] * 6,
            "home_fifa_rank": [10] * 6,
            "away_fifa_rank": [10] * 6,
            "fifa_rank_diff": [0] * 6,
            "home_fifa_points": [1200.0] * 6,
            "away_fifa_points": [1200.0] * 6,
            "fifa_points_diff": [0.0] * 6,
            "h2h_matches_prior": [0] * 6,
            "h2h_home_team_wins_prior": [0] * 6,
            "h2h_away_team_wins_prior": [0] * 6,
            "h2h_draws_prior": [0] * 6,
            "h2h_goal_diff_prior": [0] * 6,
            "home_points_last_5": [0] * 6,
            "away_points_last_5": [0] * 6,
            "home_points_per_match_last_5": [0] * 6,
            "away_points_per_match_last_5": [0] * 6,
            "home_goals_for_last_5": [0] * 6,
            "away_goals_for_last_5": [0] * 6,
            "home_goals_against_last_5": [0] * 6,
            "away_goals_against_last_5": [0] * 6,
            "home_goal_diff_last_5": [0] * 6,
            "away_goal_diff_last_5": [0] * 6,
            "home_wins_last_5": [0] * 6,
            "away_wins_last_5": [0] * 6,
            "home_draws_last_5": [0] * 6,
            "away_draws_last_5": [0] * 6,
            "home_losses_last_5": [0] * 6,
            "away_losses_last_5": [0] * 6,
            "home_points_last_10": [0] * 6,
            "away_points_last_10": [0] * 6,
            "home_points_per_match_last_10": [0] * 6,
            "away_points_per_match_last_10": [0] * 6,
            "home_goals_for_last_10": [0] * 6,
            "away_goals_for_last_10": [0] * 6,
            "home_goals_against_last_10": [0] * 6,
            "away_goals_against_last_10": [0] * 6,
            "home_goal_diff_last_10": [0] * 6,
            "away_goal_diff_last_10": [0] * 6,
            "home_wins_last_10": [0] * 6,
            "away_wins_last_10": [0] * 6,
            "home_draws_last_10": [0] * 6,
            "away_draws_last_10": [0] * 6,
            "home_losses_last_10": [0] * 6,
            "away_losses_last_10": [0] * 6,
        }
    )

    split = make_time_split(df, train_frac=0.5, val_frac=0.25)

    assert split.train["date"].max() <= split.val["date"].min()
    assert split.val["date"].max() <= split.test["date"].min()


def test_get_feature_columns_contains_expected_core_features():
    df = pd.DataFrame(
        {
            "home_team": ["A"],
            "away_team": ["B"],
            "tournament_type": ["World Cup"],
            "confederation_home": ["UEFA"],
            "confederation_away": ["UEFA"],
            "home_elo": [1500.0],
            "away_elo": [1480.0],
            "elo_diff": [20.0],
            "neutral": [True],
            "home_points_last_5": [7.0],
            "away_points_last_5": [5.0],
        }
    )

    numeric, categorical = get_feature_columns(df)

    assert "home_elo" in numeric
    assert "away_elo" in numeric
    assert "home_points_last_5" in numeric
    assert "away_points_last_5" in numeric
    assert "home_team" in categorical
    assert "away_team" in categorical
