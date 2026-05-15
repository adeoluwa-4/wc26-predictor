from pathlib import Path

import pandas as pd

from src.simulation.config import SimulationConfig
from src.simulation.team_config import groups_from_team_config, load_team_config, projected_placeholder_rows


def test_load_fixed_group_config_and_validate_shape():
    cfg = SimulationConfig()
    df = load_team_config("data/config/wc26_teams.csv", config=cfg)

    assert len(df) == 48
    assert set(df.columns) == {"group", "team", "status", "source", "notes"}
    assert set(df["group"]) == set("ABCDEFGHIJKL")
    assert df.groupby("group")["team"].count().eq(4).all()


def test_groups_from_config_builds_all_12_groups():
    cfg = SimulationConfig()
    df = load_team_config("data/config/wc26_teams.csv", config=cfg)
    groups = groups_from_team_config(df, config=cfg)

    assert set(groups.keys()) == set("ABCDEFGHIJKL")
    assert all(len(team_list) == 4 for team_list in groups.values())
    all_teams = [team for teams in groups.values() for team in teams]
    assert len(all_teams) == 48
    assert len(set(all_teams)) == 48


def test_projected_placeholder_rows_filter():
    df = load_team_config("data/config/wc26_teams.csv")
    projected = projected_placeholder_rows(df)
    if not projected.empty:
        assert (projected["status"] == "projected_placeholder").all()


def test_invalid_status_fails(tmp_path: Path):
    bad = pd.DataFrame(
        [
            {"group": "A", "team": "A1", "status": "confirmed", "source": "x", "notes": ""},
            {"group": "A", "team": "A2", "status": "bad_status", "source": "x", "notes": ""},
            {"group": "A", "team": "A3", "status": "confirmed", "source": "x", "notes": ""},
            {"group": "A", "team": "A4", "status": "confirmed", "source": "x", "notes": ""},
        ]
    )

    # Fill remaining groups to satisfy row count checks and isolate status failure.
    for group in "BCDEFGHIJKL":
        for idx in range(1, 5):
            bad.loc[len(bad)] = {
                "group": group,
                "team": f"{group}{idx}",
                "status": "confirmed",
                "source": "x",
                "notes": "",
            }

    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)

    try:
        load_team_config(path)
        assert False, "Expected ValueError for invalid status"
    except ValueError as exc:
        assert "Invalid status values" in str(exc)
