"""Head-to-head prior features with strict date safety."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def add_head_to_head_priors(matches: pd.DataFrame) -> pd.DataFrame:
    """Add prior H2H features before each match date.

    For same-date matches, all rows use only information from earlier dates.
    """
    df = matches.sort_values(["date", "match_id"]).copy()

    features = {
        "h2h_matches_prior": [],
        "h2h_home_team_wins_prior": [],
        "h2h_away_team_wins_prior": [],
        "h2h_draws_prior": [],
        "h2h_goal_diff_prior": [],
    }

    # Pair stats are stored in canonical (team1, team2) order.
    stats = defaultdict(
        lambda: {
            "matches": 0,
            "team1_wins": 0,
            "team2_wins": 0,
            "draws": 0,
            "team1_goal_diff": 0,
        }
    )

    for _, date_bucket in df.groupby("date", sort=True):
        # Read priors first (strictly before this date)
        for _, row in date_bucket.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            team1, team2 = sorted((home, away))
            pair_stats = stats[(team1, team2)]

            if home == team1:
                home_wins_prior = pair_stats["team1_wins"]
                away_wins_prior = pair_stats["team2_wins"]
                goal_diff_prior = pair_stats["team1_goal_diff"]
            else:
                home_wins_prior = pair_stats["team2_wins"]
                away_wins_prior = pair_stats["team1_wins"]
                goal_diff_prior = -pair_stats["team1_goal_diff"]

            features["h2h_matches_prior"].append(pair_stats["matches"])
            features["h2h_home_team_wins_prior"].append(home_wins_prior)
            features["h2h_away_team_wins_prior"].append(away_wins_prior)
            features["h2h_draws_prior"].append(pair_stats["draws"])
            features["h2h_goal_diff_prior"].append(goal_diff_prior)

        # Update state after all matches on this date were assigned priors.
        for _, row in date_bucket.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])

            team1, team2 = sorted((home, away))
            pair_stats = stats[(team1, team2)]
            pair_stats["matches"] += 1

            if home == team1:
                pair_stats["team1_goal_diff"] += home_score - away_score
            else:
                pair_stats["team1_goal_diff"] += away_score - home_score

            if home_score > away_score:
                if home == team1:
                    pair_stats["team1_wins"] += 1
                else:
                    pair_stats["team2_wins"] += 1
            elif away_score > home_score:
                if away == team1:
                    pair_stats["team1_wins"] += 1
                else:
                    pair_stats["team2_wins"] += 1
            else:
                pair_stats["draws"] += 1

    feature_df = pd.DataFrame({"match_id": df["match_id"].values, **features})
    merged = matches.merge(feature_df, on="match_id", how="left")
    return merged
