"""Configuration constants for the WC26 data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.utils.paths import PROCESSED_DIR, RAW_DIR

ROLLING_WINDOWS: tuple[int, ...] = (5, 10)

RESULTS_REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}

SHOOTOUTS_REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "winner",
}

FORMER_NAMES_REQUIRED_COLUMNS = {
    "current",
    "former",
    "start_date",
    "end_date",
}

FIFA_REQUIRED_COLUMNS = {"team", "total_points", "date"}

# Preferred fine-grained Elo schema. Yearly fallback is handled in loaders.
ELO_FINE_COLUMNS = {"date", "team", "elo_rating"}

ELO_CANDIDATE_FILES = (
    RAW_DIR / "elo_ratings" / "elo_ratings_historical.csv",
    RAW_DIR / "elo_ratings" / "world_football_elo_history.csv",
    RAW_DIR / "elo_ratings" / "ranking_soccer_1901-2023.csv",
    RAW_DIR / "elo_ratings" / "soccer-elo-main" / "csv" / "ranking_soccer_1901-2023.csv",
)

FIFA_CANDIDATE_FILES = (
    RAW_DIR / "fifa_rankings" / "ranking_fifa_historical.csv",
    RAW_DIR / "fifa_rankings" / "fifa_rankings_historical.csv",
)

RESULTS_CANDIDATE_FILES = (
    RAW_DIR / "international_results" / "results.csv",
    RAW_DIR / "international_results" / "international_results-master" / "results.csv",
)

SHOOTOUTS_CANDIDATE_FILES = (
    RAW_DIR / "international_results" / "shootouts.csv",
    RAW_DIR / "international_results" / "international_results-master" / "shootouts.csv",
)

FORMER_NAMES_CANDIDATE_FILES = (
    RAW_DIR / "international_results" / "former_names.csv",
    RAW_DIR / "international_results" / "international_results-master" / "former_names.csv",
)

TEAM_NAME_OVERRIDES_PATH = RAW_DIR / "reference" / "team_name_overrides.csv"
TEAM_CONFEDERATIONS_PATH = RAW_DIR / "reference" / "team_confederations.csv"

TRAINING_TABLE_PARQUET = PROCESSED_DIR / "training_matches.parquet"
TRAINING_TABLE_CSV = PROCESSED_DIR / "training_matches.csv"
VALIDATION_REPORT_PATH = PROCESSED_DIR / "validation_report.json"
UNMAPPED_TEAM_REPORT_PATH = PROCESSED_DIR / "unmapped_team_names.csv"

TOURNAMENT_IMPORTANCE = {
    "World Cup": 1.00,
    "Continental championship": 0.90,
    "World Cup qualifier": 0.75,
    "Continental qualifier": 0.65,
    "Nations League / regional competition": 0.60,
    "Friendly": 0.30,
    "Other": 0.50,
}

CONTINENTAL_FINALS_KEYWORDS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "africa cup of nations",
    "gold cup",
    "asian cup",
    "ofc nations cup",
)

CONTINENTAL_QUALIFIER_KEYWORDS = (
    "uefa euro qualification",
    "copa américa qualification",
    "copa america qualification",
    "african cup of nations qualification",
    "asian cup qualification",
    "gold cup qualification",
    "ofc nations cup qualification",
)

NATIONS_LEAGUE_KEYWORDS = (
    "nations league",
    "regional",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime config bundle for the build entrypoint."""

    rolling_windows: Sequence[int] = ROLLING_WINDOWS
    output_parquet: Path = TRAINING_TABLE_PARQUET
    output_csv: Path = TRAINING_TABLE_CSV
    output_validation: Path = VALIDATION_REPORT_PATH
    output_unmapped: Path = UNMAPPED_TEAM_REPORT_PATH
