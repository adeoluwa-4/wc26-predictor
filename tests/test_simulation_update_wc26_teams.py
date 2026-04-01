from pathlib import Path

import pandas as pd

from src.simulation import update_wc26_teams


def _make_team_config() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group in "ABCDEFGHIJKL":
        for idx in range(1, 5):
            rows.append(
                {
                    "group": group,
                    "team": f"{group}{idx}",
                    "status": "confirmed",
                    "source": "seed",
                    "notes": "",
                }
            )

    replacements = {
        "A": ("Denmark", "UEFA Play-Off D winner"),
        "B": ("Italy", "UEFA Play-Off A winner"),
        "D": ("Türkiye", "UEFA Play-Off C winner"),
        "F": ("Ukraine", "UEFA Play-Off B winner"),
        "I": ("Iraq", "FIFA Play-Off Tournament winner 2"),
        "K": ("DR Congo", "FIFA Play-Off Tournament winner 1"),
    }
    for group, (team, note) in replacements.items():
        idx = next(i for i, row in enumerate(rows) if row["group"] == group)
        rows[idx]["team"] = team
        rows[idx]["status"] = "projected_placeholder"
        rows[idx]["source"] = "user_provided"
        rows[idx]["notes"] = f"Replace {note}"

    return pd.DataFrame(rows)


def _make_qualifier_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.DataFrame(
        [
            # UEFA A
            {
                "date": "2026-03-31",
                "home_team": "Bosnia and Herzegovina",
                "away_team": "Italy",
                "home_score": 1,
                "away_score": 1,
                "tournament": "FIFA World Cup qualification",
                "city": "X",
                "country": "Y",
                "neutral": True,
            },
            # UEFA B
            {
                "date": "2026-03-31",
                "home_team": "Sweden",
                "away_team": "Poland",
                "home_score": 3,
                "away_score": 2,
                "tournament": "FIFA World Cup qualification",
                "city": "X",
                "country": "Y",
                "neutral": True,
            },
            # UEFA C
            {
                "date": "2026-03-31",
                "home_team": "Kosovo",
                "away_team": "Turkey",
                "home_score": 0,
                "away_score": 1,
                "tournament": "FIFA World Cup qualification",
                "city": "X",
                "country": "Y",
                "neutral": True,
            },
            # UEFA D
            {
                "date": "2026-03-31",
                "home_team": "Czech Republic",
                "away_team": "Denmark",
                "home_score": 2,
                "away_score": 2,
                "tournament": "FIFA World Cup qualification",
                "city": "X",
                "country": "Y",
                "neutral": True,
            },
            # FIFA playoff 1
            {
                "date": "2026-03-31",
                "home_team": "DR Congo",
                "away_team": "Jamaica",
                "home_score": 1,
                "away_score": 0,
                "tournament": "FIFA World Cup qualification",
                "city": "X",
                "country": "Y",
                "neutral": True,
            },
            # FIFA playoff 2
            {
                "date": "2026-03-31",
                "home_team": "Iraq",
                "away_team": "Bolivia",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup qualification",
                "city": "X",
                "country": "Y",
                "neutral": True,
            },
        ]
    )
    results["date"] = pd.to_datetime(results["date"])

    shootouts = pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "home_team": "Bosnia and Herzegovina",
                "away_team": "Italy",
                "winner": "Bosnia and Herzegovina",
            },
            {
                "date": "2026-03-31",
                "home_team": "Czech Republic",
                "away_team": "Denmark",
                "winner": "Czech Republic",
            },
        ]
    )
    shootouts["date"] = pd.to_datetime(shootouts["date"])
    former_names = pd.DataFrame(columns=["current", "former", "start_date", "end_date"])
    return results, shootouts, former_names


def test_resolve_wc26_team_config_replaces_placeholders(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "wc26_teams.csv"
    _make_team_config().to_csv(config_path, index=False)

    fake_results = _make_qualifier_results()
    monkeypatch.setattr(update_wc26_teams, "load_international_results", lambda: fake_results)
    monkeypatch.setattr(
        update_wc26_teams,
        "load_team_name_overrides",
        lambda: pd.DataFrame(columns=["source_name", "canonical_name", "notes"]),
    )

    updated = update_wc26_teams.resolve_wc26_team_config(config_path)
    teams = set(updated["team"].tolist())

    assert "Italy" not in teams
    assert "Bosnia and Herzegovina" in teams
    assert "Sweden" in teams
    assert "Turkey" in teams
    assert "Czech Republic" in teams
    assert (updated["status"] == "confirmed").all()
