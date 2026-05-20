from pathlib import Path

import pandas as pd

from src.data import config, loaders


def _write_bundle(bundle_dir: Path, max_date: str, marker_team: str) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {
                "date": max_date,
                "home_team": marker_team,
                "away_team": "Away",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": False,
            }
        ]
    )
    results.to_csv(bundle_dir / "results.csv", index=False)

    shootouts = pd.DataFrame(
        [
            {
                "date": max_date,
                "home_team": marker_team,
                "away_team": "Away",
                "winner": marker_team,
            }
        ]
    )
    shootouts.to_csv(bundle_dir / "shootouts.csv", index=False)

    former_names = pd.DataFrame(
        [
            {
                "current": marker_team,
                "former": marker_team,
                "start_date": "2000-01-01",
                "end_date": "2100-01-01",
            }
        ]
    )
    former_names.to_csv(bundle_dir / "former_names.csv", index=False)


def test_load_international_results_prefers_newest_bundle(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    intl_root = raw_dir / "international_results"
    _write_bundle(intl_root / "bundle_old", "2026-01-26", "OldTeam")
    _write_bundle(intl_root / "bundle_new", "2026-03-31", "NewTeam")

    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "RESULTS_CANDIDATE_FILES", tuple())

    results, shootouts, former_names = loaders.load_international_results()

    assert results["date"].max() == pd.Timestamp("2026-03-31")
    assert set(results["home_team"]) == {"NewTeam"}
    assert set(shootouts["home_team"]) == {"NewTeam"}
    assert set(former_names["current"]) == {"NewTeam"}
