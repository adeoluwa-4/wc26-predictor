"""Utilities for loading completed World Cup matches into simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.simulation.schemas import MatchFixture, SimulatedMatchResult
from src.utils.paths import PROJECT_ROOT


DEFAULT_PLAYED_MATCHES_PATH = PROJECT_ROOT / "data" / "config" / "wc26_played_matches.csv"
PLAYED_MATCH_COLUMNS = [
    "date",
    "group",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "source",
]


PlayedResultMap = dict[tuple[str, frozenset[str]], SimulatedMatchResult]


def _result_winner(home_team: str, away_team: str, home_goals: int, away_goals: int) -> str | None:
    if home_goals > away_goals:
        return home_team
    if away_goals > home_goals:
        return away_team
    return None


def _result_from_row(row: pd.Series) -> SimulatedMatchResult:
    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    home_goals = int(row["home_goals"])
    away_goals = int(row["away_goals"])
    return SimulatedMatchResult(
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        stage="group",
        group=str(row["group"]).upper(),
        winner=_result_winner(home_team, away_team, home_goals, away_goals),
        is_draw=home_goals == away_goals,
        decided_by_penalties=False,
    )


def build_played_result_map(played_matches: pd.DataFrame) -> PlayedResultMap:
    """Build group + unordered-team lookup for completed group matches."""
    if played_matches.empty:
        return {}

    missing = [col for col in PLAYED_MATCH_COLUMNS if col not in played_matches.columns]
    if missing:
        raise ValueError(f"Played matches missing required columns: {missing}")

    out: PlayedResultMap = {}
    for _, row in played_matches.iterrows():
        group = str(row["group"]).strip().upper()
        teams = frozenset({str(row["home_team"]).strip(), str(row["away_team"]).strip()})
        if len(teams) != 2:
            raise ValueError("Played match must contain two different teams")
        key = (group, teams)
        if key in out:
            raise ValueError(f"Duplicate played result for group {group}: {sorted(teams)}")
        out[key] = _result_from_row(row)
    return out


def load_played_matches(path: str | Path = DEFAULT_PLAYED_MATCHES_PATH) -> pd.DataFrame:
    """Load optional completed World Cup results file."""
    played_path = Path(path)
    if not played_path.exists():
        return pd.DataFrame(columns=PLAYED_MATCH_COLUMNS)
    df = pd.read_csv(played_path)
    missing = [col for col in PLAYED_MATCH_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Played matches file missing required columns: {missing}")
    out = df[PLAYED_MATCH_COLUMNS].copy()
    out["group"] = out["group"].astype(str).str.strip().str.upper()
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="raise").astype(int)
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="raise").astype(int)
    return out


def played_result_for_fixture(
    fixture: MatchFixture,
    played_results: Mapping[tuple[str, frozenset[str]], SimulatedMatchResult] | None,
) -> SimulatedMatchResult | None:
    """Return a completed result oriented to the requested fixture, if available."""
    if not played_results or fixture.group is None:
        return None

    key = (str(fixture.group).upper(), frozenset({fixture.home_team, fixture.away_team}))
    result = played_results.get(key)
    if result is None:
        return None

    if result.home_team == fixture.home_team and result.away_team == fixture.away_team:
        return result
    if result.home_team == fixture.away_team and result.away_team == fixture.home_team:
        return SimulatedMatchResult(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            home_goals=result.away_goals,
            away_goals=result.home_goals,
            stage=result.stage,
            group=result.group,
            winner=result.winner,
            is_draw=result.is_draw,
            decided_by_penalties=result.decided_by_penalties,
        )
    raise ValueError("Played result teams do not match requested fixture orientation")


def extract_wc26_group_matches(
    training_matches: pd.DataFrame,
    groups: Mapping[str, list[str]],
    min_date: str = "2026-06-01",
) -> pd.DataFrame:
    """Extract completed 2026 World Cup group matches from the training table."""
    df = training_matches.copy()
    if df.empty:
        return pd.DataFrame(columns=PLAYED_MATCH_COLUMNS)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    team_to_group = {
        team: group_name
        for group_name, teams in groups.items()
        for team in teams
    }

    is_world_cup = (
        df.get("tournament_type", pd.Series(index=df.index, dtype=object)).astype(str).eq("World Cup")
        | df.get("tournament", pd.Series(index=df.index, dtype=object)).astype(str).str.lower().eq("fifa world cup")
    )
    recent = df["date"] >= pd.Timestamp(min_date)
    same_group = [
        team_to_group.get(home) is not None and team_to_group.get(home) == team_to_group.get(away)
        for home, away in zip(df["home_team"], df["away_team"])
    ]

    out = df[is_world_cup & recent & pd.Series(same_group, index=df.index)].copy()
    if out.empty:
        return pd.DataFrame(columns=PLAYED_MATCH_COLUMNS)

    out["group"] = out["home_team"].map(team_to_group)
    out["source"] = "training_matches"
    out = out.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    out = out[PLAYED_MATCH_COLUMNS].sort_values(["date", "group", "home_team", "away_team"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def write_played_matches(played_matches: pd.DataFrame, path: str | Path = DEFAULT_PLAYED_MATCHES_PATH) -> Path:
    """Persist completed World Cup group matches for the simulator."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    played_matches.to_csv(out_path, index=False)
    return out_path
