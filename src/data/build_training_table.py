"""Entrypoint to build the WC26 training table from raw datasets."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

from src.data import config, head_to_head, joins, loaders, rolling_features, standardize, validation
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_directories

LOGGER = get_logger(__name__)


def _prepare_matches_base(results: pd.DataFrame) -> pd.DataFrame:
    """Prepare core match table and deterministic match IDs."""
    cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
    df = results[cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df = df.sort_values(["date", "home_team", "away_team", "tournament"]).reset_index(drop=True)
    df["match_occurrence"] = df.groupby(["date", "home_team", "away_team"]).cumcount()
    df["match_id"] = [f"M{i:07d}" for i in range(1, len(df) + 1)]

    return df


def _attach_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)
    out["away_win"] = (out["away_score"] > out["home_score"]).astype(int)
    out["draw"] = (out["home_score"] == out["away_score"]).astype(int)

    out["outcome_label"] = np.select(
        [out["home_win"] == 1, out["draw"] == 1, out["away_win"] == 1],
        ["home_win", "draw", "away_win"],
        default="draw",
    )

    out["total_goals"] = out["home_score"] + out["away_score"]
    out["goal_diff"] = out["home_score"] - out["away_score"]
    return out


def _attach_shootout_metadata(matches: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    """Attach shootout metadata to match rows without affecting pre-match features."""
    sh = shootouts[["date", "home_team", "away_team", "winner"]].copy()
    sh["date"] = pd.to_datetime(sh["date"], errors="coerce")
    sh = sh.dropna(subset=["date", "home_team", "away_team"]).copy()
    sh = sh.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    sh["match_occurrence"] = sh.groupby(["date", "home_team", "away_team"]).cumcount()

    merged = matches.merge(
        sh.rename(columns={"winner": "shootout_winner"}),
        on=["date", "home_team", "away_team", "match_occurrence"],
        how="left",
    )

    merged["went_to_shootout"] = merged["shootout_winner"].notna()
    return merged


def _classify_tournament(tournament: str) -> str:
    text = str(tournament).strip().lower()

    if "friendly" in text:
        return "Friendly"

    if "fifa world cup" in text and "qualification" in text:
        return "World Cup qualifier"

    if text == "fifa world cup":
        return "World Cup"

    if any(keyword in text for keyword in config.CONTINENTAL_QUALIFIER_KEYWORDS):
        return "Continental qualifier"

    if any(keyword in text for keyword in config.CONTINENTAL_FINALS_KEYWORDS):
        return "Continental championship"

    if any(keyword in text for keyword in config.NATIONS_LEAGUE_KEYWORDS):
        return "Nations League / regional competition"

    return "Other"


def _build_host_alias_lookup(former_names: pd.DataFrame) -> dict[str, set[str]]:
    """Build conservative team-country alias lookup using current + former names."""
    lookup: dict[str, set[str]] = {}

    for _, row in former_names.iterrows():
        current_key = standardize.normalize_key(row["current"])
        former_key = standardize.normalize_key(row["former"])
        if current_key not in lookup:
            lookup[current_key] = set()

        lookup[current_key].add(current_key)
        if former_key:
            lookup[current_key].add(former_key)

    return lookup


def _attach_context_features(
    matches: pd.DataFrame,
    former_names: pd.DataFrame,
    confederations: pd.DataFrame,
    name_map: dict[str, str],
) -> pd.DataFrame:
    """Create tournament, host, and confederation context features."""
    df = matches.copy()

    df["tournament_type"] = df["tournament"].map(_classify_tournament)
    df["is_friendly"] = (df["tournament_type"] == "Friendly")
    df["is_qualifier"] = df["tournament_type"].str.contains("qualifier", case=False)
    df["is_continental_competition"] = df["tournament_type"].isin(
        ["Continental championship", "Continental qualifier"]
    )
    df["is_world_cup"] = (df["tournament_type"] == "World Cup")
    df["tournament_importance_score"] = df["tournament_type"].map(config.TOURNAMENT_IMPORTANCE).fillna(0.50)

    host_lookup = _build_host_alias_lookup(former_names)

    def host_flag(team: str, country: str, neutral: bool) -> bool:
        if neutral:
            return False
        team_key = standardize.normalize_key(team)
        country_key = standardize.normalize_key(country)
        aliases = host_lookup.get(team_key, {team_key})
        return country_key in aliases

    df["is_host_home_country"] = [
        host_flag(team, country, neutral)
        for team, country, neutral in zip(df["home_team"], df["country"], df["neutral"])
    ]
    df["is_host_away_country"] = [
        host_flag(team, country, neutral)
        for team, country, neutral in zip(df["away_team"], df["country"], df["neutral"])
    ]

    confed_df = confederations.copy()
    if not confed_df.empty:
        confed_df["team"] = confed_df["team"].map(
            lambda name: name_map.get(standardize.normalize_key(name), standardize.normalize_team_name(str(name)))
        )
        confed_df = confed_df.drop_duplicates(subset=["team"], keep="first")

        confed_map = dict(zip(confed_df["team"], confed_df["confederation"]))
    else:
        confed_map = {}

    df["confederation_home"] = df["home_team"].map(confed_map)
    df["confederation_away"] = df["away_team"].map(confed_map)
    df["same_confederation"] = (
        df["confederation_home"].notna()
        & df["confederation_away"].notna()
        & (df["confederation_home"] == df["confederation_away"])
    )

    return df


def _write_unmapped_team_report(report: dict[str, object], output_path: str | None = None) -> pd.DataFrame:
    """Write unmapped-name diagnostics to CSV."""
    rows: list[dict[str, str]] = []

    for issue, value in report.items():
        if not isinstance(value, list):
            continue
        for team in value:
            rows.append({"issue": issue, "team": str(team)})

    unmapped_df = pd.DataFrame(rows).sort_values(["issue", "team"]).reset_index(drop=True) if rows else pd.DataFrame(columns=["issue", "team"])

    out_path = config.UNMAPPED_TEAM_REPORT_PATH if output_path is None else output_path
    unmapped_df.to_csv(out_path, index=False)
    return unmapped_df


def _ensure_output_columns(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    expected = validation.expected_output_columns(windows)
    out = df.copy()

    for col in expected:
        if col not in out.columns:
            out[col] = pd.NA

    # Keep expected columns first; preserve extras after.
    extra_cols = [col for col in out.columns if col not in expected]
    out = out[[*expected, *extra_cols]]
    return out


def build_training_table(export_csv: bool = False) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build and persist the final training table."""
    ensure_directories()

    cfg = config.PipelineConfig()

    LOGGER.info("Loading raw datasets")
    results_raw, shootouts_raw, former_names_raw = loaders.load_international_results()
    elo_raw = loaders.load_elo_history()
    fifa_raw = loaders.load_fifa_history()
    overrides = loaders.load_team_name_overrides()
    confederations = loaders.load_team_confederations()

    LOGGER.info("Standardizing team names")
    standardized = standardize.standardize_datasets(
        results=results_raw,
        shootouts=shootouts_raw,
        former_names=former_names_raw,
        elo=elo_raw,
        fifa=fifa_raw,
        overrides=overrides,
    )

    LOGGER.info("Preparing base match table")
    matches = _prepare_matches_base(standardized.results)
    matches = _attach_targets(matches)
    matches = _attach_shootout_metadata(matches, standardized.shootouts)

    LOGGER.info("Building context features")
    matches = _attach_context_features(
        matches=matches,
        former_names=standardized.former_names,
        confederations=confederations,
        name_map=standardized.name_map,
    )

    LOGGER.info("Joining Elo and FIFA rankings")
    matches, join_report = joins.join_strength_features(matches, standardized.elo, standardized.fifa)

    LOGGER.info("Building rolling form features")
    matches = rolling_features.add_rolling_features(matches, windows=cfg.rolling_windows)

    LOGGER.info("Building head-to-head priors")
    matches = head_to_head.add_head_to_head_priors(matches)

    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    matches = _ensure_output_columns(matches, cfg.rolling_windows)

    LOGGER.info("Running validation")
    report = validation.validate_training_table(matches, windows=cfg.rolling_windows, join_report=join_report)
    validation.write_validation_report(report, cfg.output_validation)

    LOGGER.info("Writing artifacts")
    cfg.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(cfg.output_parquet, index=False)
    if export_csv:
        matches.to_csv(cfg.output_csv, index=False)

    _write_unmapped_team_report(standardized.report)

    LOGGER.info(
        "Build complete | rows=%s cols=%s status=%s",
        len(matches),
        len(matches.columns),
        report["status"],
    )

    return matches, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WC26 training table")
    parser.add_argument("--export-csv", action="store_true", help="Also write training_matches.csv")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    _, report = build_training_table(export_csv=args.export_csv)
    if report["status"] != "pass":
        raise SystemExit("Validation failed. See data/processed/validation_report.json")


if __name__ == "__main__":
    main()
