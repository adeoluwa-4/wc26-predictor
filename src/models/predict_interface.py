"""Prediction interface for WC26 baseline models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.config import MODELS_DIR


class WC26Predictor:
    """Load trained artifacts and expose match-level prediction API."""

    def __init__(self, models_dir: str | Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.metadata = json.loads((self.models_dir / "model_metadata.json").read_text(encoding="utf-8"))
        self.feature_columns = self.metadata["feature_columns"]
        self.defaults = self.metadata["defaults"]

        self.team_profiles = pd.read_parquet(self.models_dir / "team_profiles.parquet")
        self.team_profiles = self.team_profiles.sort_values("team").drop_duplicates(subset=["team"], keep="last")
        self.team_profiles = self.team_profiles.set_index("team")

        self.h2h_profiles = pd.read_parquet(self.models_dir / "h2h_profiles.parquet")
        self.h2h_profiles = self.h2h_profiles.set_index(["team_a", "team_b"])

        # Speeds up Monte Carlo runs dramatically: at most 48*47 oriented matchups in a WC.
        self._prediction_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._use_fallback_model = False
        self._model_load_error: str | None = None

        try:
            self.outcome_model = joblib.load(self.models_dir / "outcome_model.joblib")
            self.home_goals_model = joblib.load(self.models_dir / "home_goals_model.joblib")
            self.away_goals_model = joblib.load(self.models_dir / "away_goals_model.joblib")
        except Exception as exc:  # pragma: no cover - environment-dependent failure path
            # Keep the app alive on deployment environments where pickle ABI/class paths differ.
            self._use_fallback_model = True
            self._model_load_error = f"{type(exc).__name__}: {exc}"
            self.outcome_model = None
            self.home_goals_model = None
            self.away_goals_model = None

    def _predict_outcome_probabilities(self, X: pd.DataFrame) -> dict[str, float]:
        model_type = self.metadata.get("outcome_model", "logistic")
        outcome_classes = list(self.metadata.get("outcome_classes", []))

        if model_type == "catboost":
            X_model = X.copy()
            for col in self.metadata.get("catboost_categorical_columns", []):
                if col in X_model.columns:
                    X_model[col] = X_model[col].fillna("Unknown").astype(str)
            outcome_proba = self.outcome_model.predict_proba(X_model)[0]
            if not outcome_classes and hasattr(self.outcome_model, "classes_"):
                outcome_classes = list(self.outcome_model.classes_)
        else:
            outcome_proba = self.outcome_model.predict_proba(X)[0]
            if not outcome_classes and hasattr(self.outcome_model, "named_steps"):
                outcome_classes = list(self.outcome_model.named_steps["model"].classes_)

        return {cls: float(prob) for cls, prob in zip(outcome_classes, outcome_proba)}

    def _team_profile(self, team: str) -> dict[str, Any]:
        if team in self.team_profiles.index:
            row = self.team_profiles.loc[team]
            return row.to_dict()

        numeric_defaults = self.defaults["numeric"]
        profile = {key: numeric_defaults[key] for key in numeric_defaults.keys()}
        profile["confederation"] = self.defaults["confederation"]
        return profile

    def _fallback_predict(self, X: pd.DataFrame) -> dict[str, float]:
        row = X.iloc[0]
        elo_diff = float(row.get("elo_diff", 0.0))
        h2h_goal_diff = float(row.get("h2h_goal_diff_prior", 0.0))
        form_diff = float(row.get("home_points_per_match_last_5", 0.0)) - float(
            row.get("away_points_per_match_last_5", 0.0)
        )

        # Heuristic strength score from pre-match signals only.
        strength_score = (elo_diff * 0.0065) + (h2h_goal_diff * 0.05) + (form_diff * 0.30)
        home_non_draw = 1.0 / (1.0 + math.exp(-strength_score))

        draw = 0.26 - min(abs(elo_diff) / 1500.0, 0.12)
        draw = max(0.12, min(0.30, draw))

        home_win = (1.0 - draw) * home_non_draw
        away_win = (1.0 - draw) * (1.0 - home_non_draw)

        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total

        home_goals = max(0.2, 1.20 + (elo_diff / 550.0) + (form_diff * 0.25))
        away_goals = max(0.2, 1.05 - (elo_diff / 650.0) - (form_diff * 0.20))

        return {
            "home_win_probability": float(home_win),
            "draw_probability": float(draw),
            "away_win_probability": float(away_win),
            "predicted_home_goals": float(home_goals),
            "predicted_away_goals": float(away_goals),
        }

    def _h2h_features(self, home_team: str, away_team: str) -> dict[str, float]:
        team_a, team_b = sorted((home_team, away_team))
        if (team_a, team_b) not in self.h2h_profiles.index:
            return {
                "h2h_matches_prior": 0.0,
                "h2h_home_team_wins_prior": 0.0,
                "h2h_away_team_wins_prior": 0.0,
                "h2h_draws_prior": 0.0,
                "h2h_goal_diff_prior": 0.0,
            }

        row = self.h2h_profiles.loc[(team_a, team_b)]
        if home_team == team_a:
            home_wins = float(row["team_a_wins"])
            away_wins = float(row["team_b_wins"])
            goal_diff = float(row["team_a_goal_diff"])
        else:
            home_wins = float(row["team_b_wins"])
            away_wins = float(row["team_a_wins"])
            goal_diff = -float(row["team_a_goal_diff"])

        return {
            "h2h_matches_prior": float(row["matches"]),
            "h2h_home_team_wins_prior": home_wins,
            "h2h_away_team_wins_prior": away_wins,
            "h2h_draws_prior": float(row["draws"]),
            "h2h_goal_diff_prior": goal_diff,
        }

    def _build_feature_row(self, home_team: str, away_team: str) -> pd.DataFrame:
        home_profile = self._team_profile(home_team)
        away_profile = self._team_profile(away_team)

        row: dict[str, Any] = {
            "home_team": home_team,
            "away_team": away_team,
            "tournament_type": "World Cup",
            "neutral": True,
            "is_friendly": False,
            "is_qualifier": False,
            "is_continental_competition": False,
            "is_world_cup": True,
            "is_host_home_country": False,
            "is_host_away_country": False,
            "tournament_importance_score": 1.0,
            "confederation_home": home_profile.get("confederation", self.defaults["confederation"]),
            "confederation_away": away_profile.get("confederation", self.defaults["confederation"]),
        }

        row["same_confederation"] = row["confederation_home"] == row["confederation_away"]

        row["home_elo"] = float(home_profile.get("elo", self.defaults["numeric"]["elo"]))
        row["away_elo"] = float(away_profile.get("elo", self.defaults["numeric"]["elo"]))
        row["elo_diff"] = row["home_elo"] - row["away_elo"]

        row["home_fifa_rank"] = float(home_profile.get("fifa_rank", self.defaults["numeric"]["fifa_rank"]))
        row["away_fifa_rank"] = float(away_profile.get("fifa_rank", self.defaults["numeric"]["fifa_rank"]))
        row["fifa_rank_diff"] = row["home_fifa_rank"] - row["away_fifa_rank"]

        row["home_fifa_points"] = float(home_profile.get("fifa_points", self.defaults["numeric"]["fifa_points"]))
        row["away_fifa_points"] = float(away_profile.get("fifa_points", self.defaults["numeric"]["fifa_points"]))
        row["fifa_points_diff"] = row["home_fifa_points"] - row["away_fifa_points"]
        row["home_fifa_available"] = float(
            pd.notna(home_profile.get("fifa_rank")) and pd.notna(home_profile.get("fifa_points"))
        )
        row["away_fifa_available"] = float(
            pd.notna(away_profile.get("fifa_rank")) and pd.notna(away_profile.get("fifa_points"))
        )
        row["fifa_pair_available"] = float(row["home_fifa_available"] * row["away_fifa_available"])

        for key, value in home_profile.items():
            if key.endswith(("_last_5", "_last_10")):
                row[f"home_{key}"] = float(value)
        for key, value in away_profile.items():
            if key.endswith(("_last_5", "_last_10")):
                row[f"away_{key}"] = float(value)

        row.update(self._h2h_features(home_team, away_team))

        for col in self.feature_columns:
            if col in row:
                continue
            if col in self.metadata["categorical_columns"]:
                row[col] = "Unknown"
            else:
                row[col] = 0.0

        return pd.DataFrame([{col: row[col] for col in self.feature_columns}])

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        if home_team == away_team:
            raise ValueError("home_team and away_team must be different")

        key = (home_team, away_team)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached

        X = self._build_feature_row(home_team=home_team, away_team=away_team)

        if self._use_fallback_model:
            out = self._fallback_predict(X)
            self._prediction_cache[key] = out
            return out

        class_probs = self._predict_outcome_probabilities(X)

        home_goals = float(max(0.0, self.home_goals_model.predict(X)[0]))
        away_goals = float(max(0.0, self.away_goals_model.predict(X)[0]))

        out = {
            "home_win_probability": class_probs.get("home_win", 0.0),
            "draw_probability": class_probs.get("draw", 0.0),
            "away_win_probability": class_probs.get("away_win", 0.0),
            "predicted_home_goals": home_goals,
            "predicted_away_goals": away_goals,
        }
        self._prediction_cache[key] = out
        return out


_DEFAULT_PREDICTOR: WC26Predictor | None = None


def predict_match(home_team: str, away_team: str, models_dir: str | Path = MODELS_DIR) -> dict[str, float]:
    """Convenience function matching the app-ready interface requirement."""
    global _DEFAULT_PREDICTOR
    models_path = Path(models_dir)

    if _DEFAULT_PREDICTOR is None or _DEFAULT_PREDICTOR.models_dir != models_path:
        _DEFAULT_PREDICTOR = WC26Predictor(models_path)

    return _DEFAULT_PREDICTOR.predict_match(home_team=home_team, away_team=away_team)
