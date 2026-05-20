"""Load and validate fixed World Cup 2026 group configuration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.simulation.config import SimulationConfig


REQUIRED_COLUMNS = ("group", "team", "status", "source", "notes")
VALID_STATUS = {"confirmed", "projected_placeholder"}


def load_team_config(path: str | Path, config: SimulationConfig | None = None) -> pd.DataFrame:
    """Load fixed group assignments from CSV and validate schema/shape."""
    cfg = config or SimulationConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Team configuration file not found: {config_path}")
    if config_path.suffix.lower() != ".csv":
        raise ValueError("Team configuration must be a CSV file")

    df = pd.read_csv(config_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Team config missing required columns: {missing}")

    out = df[list(REQUIRED_COLUMNS)].copy()
    out["group"] = out["group"].astype(str).str.strip().str.upper()
    out["team"] = out["team"].astype(str).str.strip()
    out["status"] = out["status"].astype(str).str.strip().str.lower()
    out["source"] = out["source"].fillna("").astype(str).str.strip()
    out["notes"] = out["notes"].fillna("").astype(str).str.strip()

    if out["team"].eq("").any():
        raise ValueError("Team names must be non-empty")
    if out["team"].duplicated().any():
        dupes = sorted(out.loc[out["team"].duplicated(), "team"].unique().tolist())
        raise ValueError(f"Duplicate team names found in config: {dupes}")

    bad_status = sorted(set(out["status"]) - VALID_STATUS)
    if bad_status:
        raise ValueError(f"Invalid status values in team config: {bad_status}")

    expected_groups = list(cfg.group_names[: cfg.num_groups])
    expected_group_set = set(expected_groups)
    found_group_set = set(out["group"])
    if found_group_set != expected_group_set:
        missing_groups = sorted(expected_group_set - found_group_set)
        extra_groups = sorted(found_group_set - expected_group_set)
        raise ValueError(
            f"Group labels must be exactly {expected_groups}. "
            f"Missing={missing_groups}, extra={extra_groups}"
        )

    expected_rows = cfg.num_groups * cfg.group_size
    if len(out) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in config, got {len(out)}")

    per_group = out.groupby("group")["team"].count().to_dict()
    bad_group_sizes = {g: int(n) for g, n in per_group.items() if int(n) != cfg.group_size}
    if bad_group_sizes:
        raise ValueError(f"Each group must have {cfg.group_size} teams. Found: {bad_group_sizes}")

    # Keep deterministic order by official group label.
    out["group"] = pd.Categorical(out["group"], categories=expected_groups, ordered=True)
    out = out.sort_values(["group", "team"]).reset_index(drop=True)
    out["group"] = out["group"].astype(str)

    return out


def groups_from_team_config(config_df: pd.DataFrame, config: SimulationConfig | None = None) -> dict[str, list[str]]:
    """Convert validated config rows into group->teams mapping."""
    cfg = config or SimulationConfig()
    expected_groups = list(cfg.group_names[: cfg.num_groups])

    groups: dict[str, list[str]] = {}
    for group_name in expected_groups:
        group_rows = config_df[config_df["group"] == group_name]
        teams = group_rows["team"].tolist()
        if len(teams) != cfg.group_size:
            raise ValueError(
                f"Group {group_name} must have {cfg.group_size} teams, got {len(teams)}"
            )
        groups[group_name] = teams

    return groups


def projected_placeholder_rows(config_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows marked as projected placeholders."""
    return config_df[config_df["status"] == "projected_placeholder"].copy().reset_index(drop=True)
