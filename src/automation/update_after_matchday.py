"""Refresh data and retrain models after new completed matches are available."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.build_training_table import build_training_table, _classify_tournament
from src.data.refresh_international_results import refresh_from_zip
from src.data.standardize import DEFAULT_MANUAL_ALIASES, normalize_key, normalize_team_name
from src.models.config import TRAINING_TABLE_PATH
from src.models.train_baselines import H2H_PROFILE_PATH, MODEL_METADATA_PATH, TEAM_PROFILE_PATH, train_baselines
from src.simulation.config import SimulationConfig
from src.simulation.played_matches import extract_wc26_group_matches, write_played_matches
from src.simulation.team_config import groups_from_team_config, load_team_config
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import INTERIM_DIR, PROCESSED_DIR, RAW_DIR


LOGGER = get_logger(__name__)

DEFAULT_RESULTS_ZIP_URL = "https://github.com/martj42/international_results/archive/refs/heads/master.zip"
DEFAULT_REPORT_PATH = PROCESSED_DIR / "latest_model_update_report.json"


def _download_zip(url: str, destination: Path) -> Path:
    """Download latest results archive to an interim path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        with destination.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    return destination


def _max_date(df: pd.DataFrame) -> str | None:
    if df.empty or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.dropna().empty:
        return None
    return str(dates.max().date())


def _canonical_team(name: object) -> str:
    clean = normalize_team_name(str(name))
    return DEFAULT_MANUAL_ALIASES.get(normalize_key(clean), clean)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _latest_local_results_path() -> Path | None:
    root = RAW_DIR / "international_results"
    candidates = sorted(root.rglob("results.csv")) if root.exists() else []
    if not candidates:
        return None

    ranked: list[tuple[pd.Timestamp, Path]] = []
    for path in candidates:
        try:
            df = pd.read_csv(path, usecols=["date"])
        except Exception:
            continue
        dates = pd.to_datetime(df["date"], errors="coerce")
        ranked.append((dates.max(), path))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[-1][1]


def _h2h_features(h2h: pd.DataFrame, home_team: str, away_team: str) -> dict[str, float]:
    if h2h.empty:
        return {
            "h2h_matches_prior": 0.0,
            "h2h_home_team_wins_prior": 0.0,
            "h2h_away_team_wins_prior": 0.0,
            "h2h_draws_prior": 0.0,
            "h2h_goal_diff_prior": 0.0,
        }

    team_a, team_b = sorted((home_team, away_team))
    match = h2h[(h2h["team_a"] == team_a) & (h2h["team_b"] == team_b)]
    if match.empty:
        return {
            "h2h_matches_prior": 0.0,
            "h2h_home_team_wins_prior": 0.0,
            "h2h_away_team_wins_prior": 0.0,
            "h2h_draws_prior": 0.0,
            "h2h_goal_diff_prior": 0.0,
        }

    row = match.iloc[0]
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


def _profile(team_profiles: pd.DataFrame, defaults: dict[str, Any], team: str) -> dict[str, Any]:
    match = team_profiles[team_profiles["team"] == team]
    if match.empty:
        profile = dict(defaults.get("numeric", {}))
        profile["confederation"] = defaults.get("confederation", "Unknown")
        return profile
    return match.iloc[-1].to_dict()


def _build_incremental_rows(
    existing: pd.DataFrame,
    latest_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build scored rows newer than the processed table using saved profiles."""
    if existing.empty:
        raise ValueError("Existing training table is empty; cannot build incremental fallback")

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    defaults = metadata.get("defaults", {})
    team_profiles = _load_table(TEAM_PROFILE_PATH).sort_values(["team", "date"])
    h2h_profiles = _load_table(H2H_PROFILE_PATH) if H2H_PROFILE_PATH.exists() else pd.DataFrame()

    latest = latest_results.copy()
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest["home_score"] = pd.to_numeric(latest["home_score"], errors="coerce")
    latest["away_score"] = pd.to_numeric(latest["away_score"], errors="coerce")
    latest = latest.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"]).copy()
    latest["home_team"] = latest["home_team"].map(_canonical_team)
    latest["away_team"] = latest["away_team"].map(_canonical_team)

    existing_max = pd.to_datetime(existing["date"], errors="coerce").max()
    latest = latest[latest["date"] > existing_max].copy()
    if latest.empty:
        return existing.iloc[0:0].copy()

    latest = latest.sort_values(["date", "home_team", "away_team", "tournament"]).reset_index(drop=True)
    last_match_id = int(str(existing["match_id"].iloc[-1]).replace("M", ""))

    rows: list[dict[str, Any]] = []
    for idx, raw in latest.iterrows():
        home_team = str(raw["home_team"])
        away_team = str(raw["away_team"])
        home_score = int(raw["home_score"])
        away_score = int(raw["away_score"])
        home_profile = _profile(team_profiles, defaults, home_team)
        away_profile = _profile(team_profiles, defaults, away_team)
        tournament_type = _classify_tournament(str(raw.get("tournament", "")))

        row: dict[str, Any] = {col: pd.NA for col in existing.columns}
        row.update(
            {
                "match_id": f"M{last_match_id + idx + 1:07d}",
                "date": raw["date"],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": int(home_score > away_score),
                "away_win": int(away_score > home_score),
                "draw": int(home_score == away_score),
                "outcome_label": np.select(
                    [home_score > away_score, home_score == away_score, away_score > home_score],
                    ["home_win", "draw", "away_win"],
                    default="draw",
                ).item(),
                "total_goals": home_score + away_score,
                "goal_diff": home_score - away_score,
                "tournament": raw.get("tournament", ""),
                "city": raw.get("city", ""),
                "country": raw.get("country", ""),
                "neutral": bool(raw.get("neutral", False)),
                "went_to_shootout": False,
                "shootout_winner": pd.NA,
                "match_occurrence": 0,
                "tournament_type": tournament_type,
                "is_friendly": tournament_type == "Friendly",
                "is_qualifier": "qualifier" in tournament_type.lower(),
                "is_continental_competition": tournament_type
                in {"Continental championship", "Continental qualifier"},
                "is_world_cup": tournament_type == "World Cup",
                "tournament_importance_score": 1.0 if tournament_type == "World Cup" else 0.5,
                "is_host_home_country": False,
                "is_host_away_country": False,
                "confederation_home": home_profile.get("confederation", defaults.get("confederation", "Unknown")),
                "confederation_away": away_profile.get("confederation", defaults.get("confederation", "Unknown")),
            }
        )
        row["same_confederation"] = row["confederation_home"] == row["confederation_away"]

        row["home_elo"] = home_profile.get("elo", defaults.get("numeric", {}).get("elo"))
        row["away_elo"] = away_profile.get("elo", defaults.get("numeric", {}).get("elo"))
        row["elo_diff"] = float(row["home_elo"]) - float(row["away_elo"])

        row["home_fifa_rank"] = home_profile.get("fifa_rank", defaults.get("numeric", {}).get("fifa_rank"))
        row["away_fifa_rank"] = away_profile.get("fifa_rank", defaults.get("numeric", {}).get("fifa_rank"))
        row["home_fifa_points"] = home_profile.get("fifa_points", defaults.get("numeric", {}).get("fifa_points"))
        row["away_fifa_points"] = away_profile.get("fifa_points", defaults.get("numeric", {}).get("fifa_points"))
        row["fifa_rank_diff"] = float(row["home_fifa_rank"]) - float(row["away_fifa_rank"])
        row["fifa_points_diff"] = float(row["home_fifa_points"]) - float(row["away_fifa_points"])

        for key, value in home_profile.items():
            if key.endswith(("_last_5", "_last_10")) and f"home_{key}" in row:
                row[f"home_{key}"] = value
        for key, value in away_profile.items():
            if key.endswith(("_last_5", "_last_10")) and f"away_{key}" in row:
                row[f"away_{key}"] = value

        row.update(_h2h_features(h2h_profiles, home_team, away_team))
        rows.append(row)

    return pd.DataFrame(rows, columns=existing.columns)


def _incremental_fallback_training_table(
    latest_results_path: Path,
    export_csv: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    existing = pd.read_parquet(TRAINING_TABLE_PATH)
    latest_results = pd.read_csv(latest_results_path)
    incremental = _build_incremental_rows(existing, latest_results)
    if incremental.empty:
        return existing, {
            "mode": "incremental_profile_fallback",
            "appended_rows": 0,
            "reason": "no scored rows newer than existing training table",
        }

    combined = pd.concat([existing, incremental], ignore_index=True)
    combined = combined.sort_values(["date", "match_id"]).reset_index(drop=True)
    combined.to_parquet(TRAINING_TABLE_PATH, index=False)
    if export_csv:
        combined.to_csv(TRAINING_TABLE_PATH.with_suffix(".csv"), index=False)
    return combined, {
        "mode": "incremental_profile_fallback",
        "appended_rows": int(len(incremental)),
        "first_appended_date": _max_date(incremental.sort_values("date").head(1)),
        "last_appended_date": _max_date(incremental),
    }


def run_update(
    *,
    results_zip_path: Path | None = None,
    results_zip_url: str | None = DEFAULT_RESULTS_ZIP_URL,
    export_csv: bool = True,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Refresh raw data, rebuild processed tables, extract played WC matches, and retrain."""
    raw_summary: dict[str, Any] = {}

    if results_zip_path is not None:
        raw_summary = refresh_from_zip(results_zip_path, RAW_DIR / "international_results")
    elif results_zip_url:
        zip_path = INTERIM_DIR / "international_results_latest.zip"
        _download_zip(results_zip_url, zip_path)
        raw_summary = refresh_from_zip(zip_path, RAW_DIR / "international_results")
    else:
        LOGGER.info("Skipping raw data refresh because no zip path or URL was provided")

    try:
        training_df, validation_summary = build_training_table(export_csv=export_csv)
        build_mode = "full_raw_pipeline"
    except FileNotFoundError as exc:
        output_dir = raw_summary.get("output_dir")
        results_path = Path(str(output_dir)) / "results.csv" if output_dir else _latest_local_results_path()
        if results_path is None or not results_path.exists():
            raise
        LOGGER.warning(
            "Full raw rebuild failed because a supporting raw file is missing. "
            "Using incremental profile fallback. Error: %s",
            exc,
        )
        training_df, validation_summary = _incremental_fallback_training_table(
            latest_results_path=results_path,
            export_csv=export_csv,
        )
        build_mode = "incremental_profile_fallback"

    cfg = SimulationConfig()
    team_config = load_team_config(cfg.teams_config_path, config=cfg)
    groups = groups_from_team_config(team_config, config=cfg)
    played_matches = extract_wc26_group_matches(training_df, groups=groups)
    played_path = write_played_matches(played_matches, cfg.played_matches_path)

    metadata = train_baselines()

    report = {
        "raw_refresh": raw_summary,
        "training_rows": int(len(training_df)),
        "training_max_date": _max_date(training_df),
        "build_mode": build_mode,
        "played_wc26_group_matches": int(len(played_matches)),
        "played_matches_path": str(played_path),
        "validation": validation_summary,
        "model_metrics": metadata.get("metrics", {}),
        "model_split": metadata.get("split", {}),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh latest match data, rebuild training artifacts, and retrain WC26 models."
    )
    parser.add_argument(
        "--results-zip-path",
        type=Path,
        default=None,
        help="Local zip archive containing latest international_results files.",
    )
    parser.add_argument(
        "--results-zip-url",
        default=DEFAULT_RESULTS_ZIP_URL,
        help="Remote zip URL to download when --results-zip-path is not provided.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already-present raw files and skip zip refresh.",
    )
    parser.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Only write parquet training table.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the update report JSON.",
    )
    args = parser.parse_args()

    configure_logging()
    report = run_update(
        results_zip_path=args.results_zip_path,
        results_zip_url=None if args.skip_download else args.results_zip_url,
        export_csv=not args.no_export_csv,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
