"""Stochastic match simulation built on top of model probabilities."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.simulation.config import SimulationConfig
from src.simulation.schemas import MatchFixture, SimulatedMatchResult


PredictMatchFn = Callable[[str, str], dict[str, float]]


class MatchSimulator:
    """Simulate individual matches from prediction outputs."""

    def __init__(
        self,
        predict_match_fn: PredictMatchFn,
        rng: np.random.Generator,
        config: SimulationConfig | None = None,
    ) -> None:
        self.predict_match_fn = predict_match_fn
        self.rng = rng
        self.config = config or SimulationConfig()

    def _normalize_probs(self, probs: dict[str, float]) -> tuple[float, float, float]:
        home = max(0.0, float(probs.get("home_win_probability", 0.0)))
        draw = max(0.0, float(probs.get("draw_probability", 0.0)))
        away = max(0.0, float(probs.get("away_win_probability", 0.0)))
        total = home + draw + away

        if total <= 0:
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        return home / total, draw / total, away / total

    def _sample_goals(self, mean: float) -> int:
        lam = max(0.05, float(mean))
        goals = int(self.rng.poisson(lam))
        return min(goals, self.config.goal_cap)

    def simulate_match(self, fixture: MatchFixture, knockout: bool = False) -> SimulatedMatchResult:
        """Simulate one fixture from model output and randomness."""
        pred = self.predict_match_fn(fixture.home_team, fixture.away_team)
        p_home, p_draw, p_away = self._normalize_probs(pred)

        sampled_outcome = self.rng.choice(["home_win", "draw", "away_win"], p=[p_home, p_draw, p_away])
        home_goals = self._sample_goals(float(pred.get("predicted_home_goals", 1.0)))
        away_goals = self._sample_goals(float(pred.get("predicted_away_goals", 1.0)))

        if sampled_outcome == "draw":
            shared = self._sample_goals((float(pred.get("predicted_home_goals", 1.0)) + float(pred.get("predicted_away_goals", 1.0))) / 2.0)
            home_goals = shared
            away_goals = shared
        elif sampled_outcome == "home_win" and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif sampled_outcome == "away_win" and away_goals <= home_goals:
            away_goals = home_goals + 1

        winner: str | None
        is_draw = home_goals == away_goals
        decided_by_penalties = False

        if is_draw and knockout:
            decided_by_penalties = True
            non_draw_total = p_home + p_away
            if non_draw_total <= 0:
                winner = fixture.home_team if self.rng.random() < 0.5 else fixture.away_team
            else:
                winner = self.rng.choice(
                    [fixture.home_team, fixture.away_team],
                    p=[p_home / non_draw_total, p_away / non_draw_total],
                )
        elif home_goals > away_goals:
            winner = fixture.home_team
        elif away_goals > home_goals:
            winner = fixture.away_team
        else:
            winner = None

        return SimulatedMatchResult(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            stage=fixture.stage,
            group=fixture.group,
            winner=winner,
            is_draw=is_draw,
            decided_by_penalties=decided_by_penalties,
        )
