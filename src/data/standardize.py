"""Team/country name standardization for cross-dataset joins."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_MANUAL_ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "korea, republic of": "South Korea",
    "ir iran": "Iran",
    "china pr": "China",
    "cabo verde": "Cape Verde",
    "congo dr": "DR Congo",
    "curacao": "Curaçao",
}


@dataclass
class StandardizationResult:
    """Container for standardized tables and name mapping report."""

    results: pd.DataFrame
    shootouts: pd.DataFrame
    former_names: pd.DataFrame
    elo: pd.DataFrame
    fifa: pd.DataFrame
    name_map: dict[str, str]
    report: dict[str, object]


def normalize_key(value: object) -> str:
    """Create a stable key for name matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u2019", "'")
    text = WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def normalize_team_name(name: str) -> str:
    """Public helper for lightweight text cleanup."""
    return WHITESPACE_RE.sub(" ", str(name)).strip()


def _build_name_map(former_names: pd.DataFrame, overrides: pd.DataFrame | None = None) -> dict[str, str]:
    """Build canonical map from former names + overrides + safe manual aliases."""
    mapping: dict[str, str] = {}

    for _, row in former_names.iterrows():
        current = normalize_team_name(row["current"])
        former = normalize_team_name(row["former"])
        if current:
            mapping[normalize_key(current)] = current
        if former:
            mapping[normalize_key(former)] = current

    if overrides is not None and not overrides.empty:
        for _, row in overrides.iterrows():
            source_name = normalize_team_name(row["source_name"])
            canonical_name = normalize_team_name(row["canonical_name"])
            if source_name and canonical_name:
                mapping[normalize_key(source_name)] = canonical_name
                mapping[normalize_key(canonical_name)] = canonical_name

    for alias, canonical in DEFAULT_MANUAL_ALIASES.items():
        if normalize_key(alias) not in mapping:
            mapping[normalize_key(alias)] = canonical

    return mapping


def _standardize_column(series: pd.Series, name_map: dict[str, str]) -> tuple[pd.Series, set[str]]:
    unmapped: set[str] = set()

    def convert(value: object) -> object:
        if pd.isna(value):
            return value
        clean = normalize_team_name(str(value))
        key = normalize_key(clean)
        canonical = name_map.get(key)
        if canonical is None:
            unmapped.add(clean)
            return clean
        return canonical

    return series.map(convert), unmapped


def _apply_name_map(df: pd.DataFrame, columns: Iterable[str], name_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    out = df.copy()
    unmapped_per_column: dict[str, set[str]] = {}

    for col in columns:
        if col not in out.columns:
            continue
        standardized, unmapped = _standardize_column(out[col], name_map)
        out[col] = standardized
        unmapped_per_column[col] = unmapped

    return out, unmapped_per_column


def build_team_alias_lookup(former_names: pd.DataFrame) -> dict[str, set[str]]:
    """Build normalized alias lookup used for conservative host-country flags."""
    alias_lookup: dict[str, set[str]] = {}

    for _, row in former_names.iterrows():
        current = normalize_team_name(row["current"])
        former = normalize_team_name(row["former"])
        current_key = normalize_key(current)

        if current_key not in alias_lookup:
            alias_lookup[current_key] = set()

        alias_lookup[current_key].add(current_key)
        alias_lookup[current_key].add(normalize_key(former))

    return alias_lookup


def standardize_datasets(
    results: pd.DataFrame,
    shootouts: pd.DataFrame,
    former_names: pd.DataFrame,
    elo: pd.DataFrame,
    fifa: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
) -> StandardizationResult:
    """Standardize names across datasets and return mapping diagnostics."""
    name_map = _build_name_map(former_names, overrides)

    results_std, results_unmapped = _apply_name_map(results, ["home_team", "away_team"], name_map)
    shootouts_std, shootouts_unmapped = _apply_name_map(shootouts, ["home_team", "away_team", "winner"], name_map)
    former_std, _ = _apply_name_map(former_names, ["current", "former"], name_map)
    elo_std, elo_unmapped = _apply_name_map(elo, ["team"], name_map)
    fifa_std, fifa_unmapped = _apply_name_map(fifa, ["team"], name_map)

    results_teams = set(results_std["home_team"]).union(set(results_std["away_team"]))
    elo_teams = set(elo_std["team"])
    fifa_teams = set(fifa_std["team"])

    report = {
        "name_map_size": len(name_map),
        "results_unmapped_home": sorted(results_unmapped.get("home_team", set())),
        "results_unmapped_away": sorted(results_unmapped.get("away_team", set())),
        "elo_unmapped": sorted(elo_unmapped.get("team", set())),
        "fifa_unmapped": sorted(fifa_unmapped.get("team", set())),
        "results_not_in_elo": sorted(results_teams - elo_teams),
        "results_not_in_fifa": sorted(results_teams - fifa_teams),
        "elo_not_in_results": sorted(elo_teams - results_teams),
        "fifa_not_in_results": sorted(fifa_teams - results_teams),
        "shootouts_unmapped": sorted(
            set().union(
                shootouts_unmapped.get("home_team", set()),
                shootouts_unmapped.get("away_team", set()),
                shootouts_unmapped.get("winner", set()),
            )
        ),
    }

    return StandardizationResult(
        results=results_std,
        shootouts=shootouts_std,
        former_names=former_std,
        elo=elo_std,
        fifa=fifa_std,
        name_map=name_map,
        report=report,
    )
