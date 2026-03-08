"""Team configuration loading and confederation-constrained group drawing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.config import SimulationConfig


UEFA_MAX_PER_GROUP = 2
NON_UEFA_MAX_PER_GROUP = 1
REQUIRED_COLUMNS = ("team", "confederation", "pot")


def _confed_limit(confederation: str) -> int:
    return UEFA_MAX_PER_GROUP if confederation == "UEFA" else NON_UEFA_MAX_PER_GROUP


def load_team_config(path: str | Path, config: SimulationConfig | None = None) -> pd.DataFrame:
    """Load and validate the team configuration CSV."""
    cfg = config or SimulationConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Team configuration file not found: {config_path}")

    if config_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV team configuration is supported in MVP pipeline")

    df = pd.read_csv(config_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Team config missing required columns: {missing}")

    out = df.copy()
    out["team"] = out["team"].astype(str).str.strip()
    out["confederation"] = out["confederation"].astype(str).str.strip().str.upper()
    out["pot"] = pd.to_numeric(out["pot"], errors="coerce").astype("Int64")
    if "group" in out.columns:
        out["group"] = out["group"].astype("string").str.strip().str.upper()
        out.loc[out["group"].isin(["", "NAN", "NONE"]), "group"] = pd.NA
    else:
        out["group"] = pd.NA

    expected_teams = cfg.num_groups * cfg.group_size
    if len(out) != expected_teams:
        raise ValueError(f"Expected {expected_teams} teams, found {len(out)}")

    if out["team"].eq("").any() or out["team"].isna().any():
        raise ValueError("Team names must be non-empty")
    if out["team"].duplicated().any():
        dupes = sorted(out.loc[out["team"].duplicated(), "team"].unique().tolist())
        raise ValueError(f"Duplicate team names in config: {dupes}")

    if out["confederation"].eq("").any() or out["confederation"].isna().any():
        raise ValueError("Confederation values must be non-empty")

    if out["pot"].isna().any():
        raise ValueError("All pot values must be integers")

    valid_pots = set(range(1, cfg.group_size + 1))
    present_pots = set(out["pot"].astype(int).unique().tolist())
    if present_pots != valid_pots:
        raise ValueError(f"Pots must be exactly {sorted(valid_pots)}, got {sorted(present_pots)}")

    pot_counts = out.groupby("pot")["team"].count().to_dict()
    for pot in valid_pots:
        if int(pot_counts.get(pot, 0)) != cfg.num_groups:
            raise ValueError(
                f"Pot {pot} must contain exactly {cfg.num_groups} teams, got {pot_counts.get(pot, 0)}"
            )

    known_groups = set(cfg.group_names[: cfg.num_groups])
    assigned_groups = set(out["group"].dropna().tolist())
    unknown_groups = assigned_groups - known_groups
    if unknown_groups:
        raise ValueError(f"Unknown groups in team config: {sorted(unknown_groups)}")

    return out


def build_groups_from_team_config(
    teams_df: pd.DataFrame,
    rng: np.random.Generator,
    config: SimulationConfig | None = None,
) -> dict[str, list[str]]:
    """Build group assignments from team config, drawing randomly when group is omitted."""
    cfg = config or SimulationConfig()
    cfg.validate()

    group_names = list(cfg.group_names[: cfg.num_groups])
    groups: dict[str, list[str]] = {name: [] for name in group_names}
    group_confeds: dict[str, dict[str, int]] = {name: {} for name in group_names}
    group_pots: dict[str, set[int]] = {name: set() for name in group_names}

    def can_place(group: str, confederation: str, pot: int) -> bool:
        if len(groups[group]) >= cfg.group_size:
            return False
        if pot in group_pots[group]:
            return False
        current = group_confeds[group].get(confederation, 0)
        if current + 1 > _confed_limit(confederation):
            return False
        return True

    def place(group: str, row: dict[str, object]) -> None:
        team = str(row["team"])
        conf = str(row["confederation"])
        pot = int(row["pot"])
        groups[group].append(team)
        group_pots[group].add(pot)
        group_confeds[group][conf] = group_confeds[group].get(conf, 0) + 1

    def unplace(group: str, row: dict[str, object]) -> None:
        team = str(row["team"])
        conf = str(row["confederation"])
        pot = int(row["pot"])
        groups[group].remove(team)
        group_pots[group].remove(pot)
        group_confeds[group][conf] -= 1
        if group_confeds[group][conf] == 0:
            del group_confeds[group][conf]

    assigned = teams_df[teams_df["group"].notna()].copy()
    for _, row in assigned.iterrows():
        group = str(row["group"])
        conf = str(row["confederation"])
        pot = int(row["pot"])
        if not can_place(group, conf, pot):
            raise ValueError(
                f"Invalid pre-assigned group placement for team={row['team']}, "
                f"group={group}, confederation={conf}, pot={pot}"
            )
        place(group, row.to_dict())

    remaining_rows = teams_df[teams_df["group"].isna()].copy()
    remaining = [row._asdict() for row in remaining_rows.itertuples(index=False, name="TeamRow")]

    def solve(remaining_items: list[dict[str, object]]) -> bool:
        if not remaining_items:
            return True

        best_idx = -1
        best_options: list[str] | None = None
        for idx, row in enumerate(remaining_items):
            conf = str(row["confederation"])
            pot = int(row["pot"])
            options = [g for g in group_names if can_place(g, conf, pot)]
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_idx = idx
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_options is not None
        row = remaining_items.pop(best_idx)
        order = list(rng.permutation(best_options))
        for group in order:
            place(group, row)
            if solve(remaining_items):
                return True
            unplace(group, row)
        remaining_items.insert(best_idx, row)
        return False

    if not solve(remaining):
        raise ValueError("Could not build valid groups from team config and constraints")

    for group_name, team_list in groups.items():
        if len(team_list) != cfg.group_size:
            raise ValueError(f"Group {group_name} has {len(team_list)} teams, expected {cfg.group_size}")

    assigned_teams = [team for team_list in groups.values() for team in team_list]
    if len(set(assigned_teams)) != cfg.num_groups * cfg.group_size:
        raise ValueError("Built groups contain duplicate or missing teams")

    return groups
