from pathlib import Path

import pandas as pd

from src.platform import pipeline
from src.platform.pipeline import (
    PipelineConfig,
    Warehouse,
    create_immutable_snapshot,
    stable_match_key,
    validate_and_quarantine,
)


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        warehouse_path=tmp_path / "warehouse.duckdb",
        snapshot_root=tmp_path / "snapshots",
        quarantine_root=tmp_path / "quarantine",
        artifacts_root=tmp_path / "published",
        dbt_project_dir=tmp_path / "warehouse",
    )


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "home_team": " USA ",
                "away_team": "Mexico",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "Austin",
                "country": "United States",
                "neutral": False,
            },
            {
                "date": "2025-02-01",
                "home_team": "Canada",
                "away_team": "Mexico",
                "home_score": 2,
                "away_score": 2,
                "tournament": "Friendly",
                "city": "Toronto",
                "country": "Canada",
                "neutral": False,
            },
        ]
    )


def test_stable_match_key_ignores_corrected_score():
    original = _matches().iloc[0].to_dict()
    corrected = dict(original, home_score=3, away_score=1)
    assert stable_match_key(original) == stable_match_key(corrected)


def test_snapshot_is_immutable_and_manifested(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    source = tmp_path / "results.csv"
    _matches().to_csv(source, index=False)
    fixed = pd.Timestamp("2026-10-23T12:00:00Z").to_pydatetime()
    monkeypatch.setattr(pipeline, "utc_now", lambda: fixed)

    manifest = create_immutable_snapshot(source, config)

    assert Path(manifest["snapshot_path"]).exists()
    assert Path(manifest["snapshot_path"] + ".manifest.json").exists()
    try:
        create_immutable_snapshot(source, config)
        assert False, "Expected immutable snapshot collision to fail"
    except FileExistsError:
        pass


def test_invalid_rows_are_quarantined(tmp_path: Path):
    config = _config(tmp_path)
    frame = _matches()
    frame.loc[1, "home_score"] = -1

    valid, invalid = validate_and_quarantine(frame, "run-1", config)

    assert len(valid) == 1
    assert len(invalid) == 1
    assert "home_score" in invalid.iloc[0]["quarantine_reason"]
    assert (config.quarantine_root / "run-1.parquet").exists()


def test_incremental_load_is_idempotent_and_upserts_corrections(tmp_path: Path):
    config = _config(tmp_path)
    valid, _ = validate_and_quarantine(_matches(), "run-1", config)
    warehouse = Warehouse(config.warehouse_path)
    try:
        first = warehouse.load_matches(valid, config.source_name, "snapshot-1")
        second = warehouse.load_matches(valid, config.source_name, "snapshot-1")
        corrected = valid.copy()
        corrected.loc[1, "home_score"] = 4
        corrected.loc[1, "away_score"] = 1
        third = warehouse.load_matches(corrected, config.source_name, "snapshot-2")
        rows = warehouse.connection.execute(
            "select match_key, home_score, away_score from raw_matches order by date"
        ).fetchall()
    finally:
        warehouse.close()

    assert first["rows_loaded"] == 2
    assert second["rows_loaded"] == 1
    assert third["rows_loaded"] == 1
    assert len(rows) == 2
    assert rows[1][1:] == (4, 1)


def test_backfill_filters_requested_date_range(tmp_path: Path):
    config = _config(tmp_path)
    valid, _ = validate_and_quarantine(_matches(), "run-1", config)
    warehouse = Warehouse(config.warehouse_path)
    try:
        result = warehouse.load_matches(
            valid,
            config.source_name,
            "snapshot-1",
            backfill_start="2025-02-01",
            backfill_end="2025-02-28",
        )
        count = warehouse.connection.execute("select count(*) from raw_matches").fetchone()[0]
    finally:
        warehouse.close()

    assert result["rows_loaded"] == 1
    assert count == 1
