"""Raw dataset loaders and schema checks for WC26 pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data import config

LOGGER = logging.getLogger(__name__)


def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with UTF-8 first, then latin-1 fallback."""
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not read CSV file: {path}")


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def _find_existing_path(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback by basename search under raw root.
    for candidate in candidates:
        found = sorted(config.RAW_DIR.rglob(candidate.name))
        if found:
            return found[0]

    raise FileNotFoundError(
        "Unable to find required file. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _is_valid_raw_csv(path: Path) -> bool:
    return path.is_file() and "__MACOSX" not in path.parts


def _discover_international_results_bundles() -> list[tuple[Path, Path, Path]]:
    """Discover candidate (results, shootouts, former_names) bundles.

    Any directory containing all three CSVs is treated as a candidate.
    """
    root = config.RAW_DIR / "international_results"
    bundles: list[tuple[Path, Path, Path]] = []
    seen: set[tuple[Path, Path, Path]] = set()

    if root.exists():
        for results_path in root.rglob("results.csv"):
            if not _is_valid_raw_csv(results_path):
                continue
            bundle_dir = results_path.parent
            shootouts_path = bundle_dir / "shootouts.csv"
            former_names_path = bundle_dir / "former_names.csv"
            if not (_is_valid_raw_csv(shootouts_path) and _is_valid_raw_csv(former_names_path)):
                continue
            bundle = (results_path, shootouts_path, former_names_path)
            if bundle not in seen:
                seen.add(bundle)
                bundles.append(bundle)

    # Keep backward compatibility with explicit candidate paths.
    for results_path in config.RESULTS_CANDIDATE_FILES:
        if not _is_valid_raw_csv(results_path):
            continue
        bundle_dir = results_path.parent
        shootouts_path = bundle_dir / "shootouts.csv"
        former_names_path = bundle_dir / "former_names.csv"
        if not (_is_valid_raw_csv(shootouts_path) and _is_valid_raw_csv(former_names_path)):
            continue
        bundle = (results_path, shootouts_path, former_names_path)
        if bundle not in seen:
            seen.add(bundle)
            bundles.append(bundle)

    return bundles


def _results_max_date(path: Path) -> pd.Timestamp:
    """Return max match date from a results CSV for bundle ranking."""
    df = _read_csv_with_fallback(path, usecols=["date"])
    dates = pd.to_datetime(df["date"], errors="coerce")
    return dates.max()


def _select_newest_results_bundle(
    bundles: list[tuple[Path, Path, Path]],
) -> tuple[Path, Path, Path]:
    if not bundles:
        raise FileNotFoundError(
            "No valid international results bundle found under "
            f"{config.RAW_DIR / 'international_results'}"
        )

    ranked: list[tuple[pd.Timestamp, tuple[Path, Path, Path]]] = []
    for bundle in bundles:
        max_date = _results_max_date(bundle[0])
        ranked.append((max_date, bundle))

    ranked.sort(key=lambda item: item[0])
    selected_date, selected_bundle = ranked[-1]
    LOGGER.info(
        "Selected international results bundle: %s (max_date=%s)",
        selected_bundle[0].parent,
        selected_date.date() if pd.notna(selected_date) else "NaT",
    )
    return selected_bundle


def load_international_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load results, shootouts, and former names tables."""
    bundles = _discover_international_results_bundles()
    results_path, shootouts_path, former_names_path = _select_newest_results_bundle(bundles)

    results = _read_csv_with_fallback(results_path)
    shootouts = _read_csv_with_fallback(shootouts_path)
    former_names = _read_csv_with_fallback(former_names_path)

    _require_columns(results, config.RESULTS_REQUIRED_COLUMNS, "results")
    _require_columns(shootouts, config.SHOOTOUTS_REQUIRED_COLUMNS, "shootouts")
    _require_columns(former_names, config.FORMER_NAMES_REQUIRED_COLUMNS, "former_names")

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
    former_names["start_date"] = pd.to_datetime(former_names["start_date"], errors="coerce")
    former_names["end_date"] = pd.to_datetime(former_names["end_date"], errors="coerce")

    if results["date"].isna().any():
        bad_count = int(results["date"].isna().sum())
        raise ValueError(f"results has {bad_count} rows with invalid dates")

    if shootouts["date"].isna().any():
        bad_count = int(shootouts["date"].isna().sum())
        raise ValueError(f"shootouts has {bad_count} rows with invalid dates")

    neutral_map = {
        "TRUE": True,
        "FALSE": False,
        "true": True,
        "false": False,
        True: True,
        False: False,
        1: True,
        0: False,
    }
    results["neutral"] = results["neutral"].map(neutral_map).fillna(False).astype(bool)

    return results, shootouts, former_names


def _clean_elo_history(raw_elo: pd.DataFrame) -> pd.DataFrame:
    """Normalize Elo history to date/team/elo_rating schema.

    Supports:
    - Fine-grained snapshots: date + team + rating-like column
    - Yearly fallback snapshots: year + team + rating
    """
    df = raw_elo.copy()
    columns_lower = {col.lower(): col for col in df.columns}

    rename_map: dict[str, str] = {}

    # Team column detection
    for key in ("team", "country", "nation"):
        if key in columns_lower:
            rename_map[columns_lower[key]] = "team"
            break

    # Rating column detection
    for key in ("elo_rating", "elo", "rating", "points"):
        if key in columns_lower:
            rename_map[columns_lower[key]] = "elo_rating"
            break

    # Date/year detection
    has_date = any(k in columns_lower for k in ("date", "ranking_date", "snapshot_date"))
    has_year = "year" in columns_lower

    if has_date:
        for key in ("date", "ranking_date", "snapshot_date"):
            if key in columns_lower:
                rename_map[columns_lower[key]] = "date"
                break
    elif has_year:
        rename_map[columns_lower["year"]] = "year"

    df = df.rename(columns=rename_map)

    if "team" not in df.columns or "elo_rating" not in df.columns:
        raise ValueError(
            "Elo file must include team and rating columns. "
            f"Detected columns: {list(raw_elo.columns)}"
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        granularity = "dated"
    elif "year" in df.columns:
        year_numeric = pd.to_numeric(df["year"], errors="coerce")
        df["date"] = pd.to_datetime(year_numeric.astype("Int64").astype(str) + "-12-31", errors="coerce")
        granularity = "year_end_fallback"
    else:
        raise ValueError("Elo file must provide either a date column or a year column")

    df["elo_rating"] = pd.to_numeric(df["elo_rating"], errors="coerce")

    df = df[["date", "team", "elo_rating"]].dropna(subset=["date", "team", "elo_rating"]).copy()
    df["team"] = df["team"].astype(str).str.strip()
    df["elo_granularity"] = granularity

    # Keep the most recent duplicated row for a team/date pair.
    df = (
        df.sort_values(["team", "date", "elo_rating"])
        .drop_duplicates(subset=["team", "date"], keep="last")
        .sort_values(["team", "date"]) 
        .reset_index(drop=True)
    )

    return df


def load_elo_history() -> pd.DataFrame:
    """Load and clean Elo history table."""
    elo_path = _find_existing_path(config.ELO_CANDIDATE_FILES)
    elo_raw = _read_csv_with_fallback(elo_path)
    elo_clean = _clean_elo_history(elo_raw)
    return elo_clean


def load_fifa_history() -> pd.DataFrame:
    """Load and clean FIFA rankings history table."""
    fifa_path = _find_existing_path(config.FIFA_CANDIDATE_FILES)
    fifa_raw = _read_csv_with_fallback(fifa_path)

    _require_columns(fifa_raw, config.FIFA_REQUIRED_COLUMNS, "fifa_rankings")

    fifa = fifa_raw.rename(columns={"total_points": "fifa_points"}).copy()
    fifa["date"] = pd.to_datetime(fifa["date"], errors="coerce")
    fifa["fifa_points"] = pd.to_numeric(fifa["fifa_points"], errors="coerce")
    fifa["team"] = fifa["team"].astype(str).str.strip()

    fifa = fifa.dropna(subset=["date", "team", "fifa_points"]).copy()

    fifa = (
        fifa.sort_values(["date", "team", "fifa_points"], ascending=[True, True, False])
        .drop_duplicates(subset=["date", "team"], keep="first")
        .reset_index(drop=True)
    )

    fifa["fifa_rank"] = (
        fifa.groupby("date")["fifa_points"]
        .rank(method="first", ascending=False)
        .astype("int64")
    )

    fifa = fifa[["date", "team", "fifa_rank", "fifa_points"]].sort_values(["team", "date"]).reset_index(drop=True)
    return fifa


def load_team_name_overrides() -> pd.DataFrame:
    """Load optional manual team-name overrides."""
    path = config.TEAM_NAME_OVERRIDES_PATH
    if not path.exists():
        return pd.DataFrame(columns=["source_name", "canonical_name", "notes"])

    overrides = _read_csv_with_fallback(path)
    required = {"source_name", "canonical_name"}
    _require_columns(overrides, required, "team_name_overrides")
    return overrides


def load_team_confederations() -> pd.DataFrame:
    """Load optional team-to-confederation mapping table."""
    path = config.TEAM_CONFEDERATIONS_PATH
    if not path.exists():
        return pd.DataFrame(columns=["team", "confederation"])

    confed = _read_csv_with_fallback(path)

    if "canonical_team" in confed.columns and "team" not in confed.columns:
        confed = confed.rename(columns={"canonical_team": "team"})

    required = {"team", "confederation"}
    _require_columns(confed, required, "team_confederations")

    confed = confed[["team", "confederation"]].copy()
    confed["team"] = confed["team"].astype(str).str.strip()
    confed["confederation"] = confed["confederation"].astype(str).str.strip()
    confed = confed.dropna().drop_duplicates(subset=["team"], keep="first")
    return confed
