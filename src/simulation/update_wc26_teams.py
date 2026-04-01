"""Resolve WC26 projected qualifier slots using latest qualification results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loaders import load_international_results, load_team_name_overrides
from src.data.standardize import DEFAULT_MANUAL_ALIASES, normalize_key, normalize_team_name
from src.simulation.config import SimulationConfig
from src.simulation.team_config import load_team_config
from src.utils.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PlayoffSlot:
    group: str
    slot_label: str
    home_team: str
    away_team: str


PLAYOFF_SLOTS: tuple[PlayoffSlot, ...] = (
    PlayoffSlot("B", "UEFA Play-Off A winner", "Bosnia and Herzegovina", "Italy"),
    PlayoffSlot("F", "UEFA Play-Off B winner", "Sweden", "Poland"),
    PlayoffSlot("D", "UEFA Play-Off C winner", "Kosovo", "Turkey"),
    PlayoffSlot("A", "UEFA Play-Off D winner", "Czech Republic", "Denmark"),
    PlayoffSlot("K", "FIFA Play-Off Tournament winner 1", "DR Congo", "Jamaica"),
    PlayoffSlot("I", "FIFA Play-Off Tournament winner 2", "Iraq", "Bolivia"),
)


def _build_canonical_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {normalize_key(k): v for k, v in DEFAULT_MANUAL_ALIASES.items()}
    overrides = load_team_name_overrides()
    if not overrides.empty:
        for _, row in overrides.iterrows():
            source_name = normalize_team_name(str(row["source_name"]))
            canonical_name = normalize_team_name(str(row["canonical_name"]))
            if source_name and canonical_name:
                mapping[normalize_key(source_name)] = canonical_name
                mapping[normalize_key(canonical_name)] = canonical_name
    return mapping


def _canonicalize(name: str, canonical_map: dict[str, str]) -> str:
    key = normalize_key(name)
    if key in canonical_map:
        return canonical_map[key]
    return normalize_team_name(name)


def _qualification_matches(results: pd.DataFrame) -> pd.DataFrame:
    out = results[
        results["tournament"].astype(str).str.contains("FIFA World Cup qualification", case=False, na=False)
    ].copy()
    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def _resolve_match_winner(
    qualification_results: pd.DataFrame,
    shootouts: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> tuple[str, pd.Series]:
    pair_matches = qualification_results[
        (
            (qualification_results["home_team"] == home_team)
            & (qualification_results["away_team"] == away_team)
        )
        | (
            (qualification_results["home_team"] == away_team)
            & (qualification_results["away_team"] == home_team)
        )
    ].copy()
    if pair_matches.empty:
        raise ValueError(f"No qualification match found for pair: {home_team} vs {away_team}")

    match_row = pair_matches.sort_values("date").iloc[-1]
    home_score = int(match_row["home_score"])
    away_score = int(match_row["away_score"])

    if home_score > away_score:
        return str(match_row["home_team"]), match_row
    if away_score > home_score:
        return str(match_row["away_team"]), match_row

    shoot = shootouts[
        (shootouts["date"] == match_row["date"])
        & (shootouts["home_team"] == match_row["home_team"])
        & (shootouts["away_team"] == match_row["away_team"])
    ]
    if shoot.empty:
        # Fallback for potential orientation mismatch.
        shoot = shootouts[
            (shootouts["date"] == match_row["date"])
            & (shootouts["home_team"] == match_row["away_team"])
            & (shootouts["away_team"] == match_row["home_team"])
        ]
    if shoot.empty:
        raise ValueError(
            "Tie match has no shootout winner in shootouts.csv for "
            f"{match_row['home_team']} vs {match_row['away_team']} on {match_row['date'].date()}"
        )

    winner = str(shoot.iloc[-1]["winner"])
    return winner, match_row


def _row_index_for_slot(df: pd.DataFrame, slot: PlayoffSlot) -> int:
    group_rows = df[df["group"] == slot.group]
    if group_rows.empty:
        raise ValueError(f"Group {slot.group} not found in team config")

    candidate = group_rows[
        group_rows["notes"].astype(str).str.contains(slot.slot_label, case=False, na=False)
        | group_rows["source"].astype(str).str.contains(slot.slot_label, case=False, na=False)
    ]
    if len(candidate) == 1:
        return int(candidate.index[0])

    placeholder = group_rows[group_rows["status"] == "projected_placeholder"]
    if len(placeholder) == 1:
        return int(placeholder.index[0])

    raise ValueError(
        f"Could not uniquely locate config row for group={slot.group}, slot={slot.slot_label}. "
        f"Matched_by_label={len(candidate)}, placeholders_in_group={len(placeholder)}"
    )


def resolve_wc26_team_config(config_path: Path) -> pd.DataFrame:
    """Resolve projected slots from latest qualification results and write config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Team config file not found: {config_path}")

    results, shootouts, _ = load_international_results()
    qual_results = _qualification_matches(results)
    canonical_map = _build_canonical_name_map()

    df = pd.read_csv(config_path)
    required_columns = {"group", "team", "status", "source", "notes"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Team config missing columns: {missing}")

    for slot in PLAYOFF_SLOTS:
        winner_raw, match_row = _resolve_match_winner(
            qualification_results=qual_results,
            shootouts=shootouts,
            home_team=slot.home_team,
            away_team=slot.away_team,
        )
        winner = _canonicalize(winner_raw, canonical_map)
        row_idx = _row_index_for_slot(df, slot)

        df.loc[row_idx, "team"] = winner
        df.loc[row_idx, "status"] = "confirmed"
        df.loc[row_idx, "source"] = f"qualification_results_auto:{slot.slot_label}"
        df.loc[row_idx, "notes"] = (
            f"Resolved {slot.slot_label} on {pd.Timestamp(match_row['date']).date()} "
            f"({match_row['home_team']} {int(match_row['home_score'])}-{int(match_row['away_score'])} "
            f"{match_row['away_team']})"
        )
        LOGGER.info("Resolved %s => %s", slot.slot_label, winner)

    df["team"] = df["team"].map(lambda value: _canonicalize(str(value), canonical_map))
    if df["team"].duplicated().any():
        duplicates = sorted(df.loc[df["team"].duplicated(keep=False), "team"].unique().tolist())
        raise ValueError(f"Duplicate teams found after resolution: {duplicates}")

    df.to_csv(config_path, index=False)

    validated = load_team_config(config_path, config=SimulationConfig())
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve WC26 projected qualifier slots from latest results.")
    parser.add_argument(
        "--config-path",
        default="data/config/wc26_teams.csv",
        help="Path to wc26 team config CSV",
    )
    args = parser.parse_args()

    configure_logging()
    validated = resolve_wc26_team_config(Path(args.config_path))
    status_counts = validated["status"].value_counts().to_dict()
    LOGGER.info("Updated %s rows in %s", len(validated), args.config_path)
    LOGGER.info("Status counts: %s", status_counts)


if __name__ == "__main__":
    main()
