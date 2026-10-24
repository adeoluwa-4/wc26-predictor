"""Incremental, idempotent DuckDB loading for the WC26 data platform."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema

from src.data.build_training_table import build_training_table
from src.data.standardize import normalize_team_name
from src.models.train_baselines import train_baselines
from src.utils.paths import PROJECT_ROOT, RAW_DIR


RAW_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

MATCH_SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False, coerce=True),
        "home_team": Column(str, Check.str_length(min_value=1), nullable=False, coerce=True),
        "away_team": Column(str, Check.str_length(min_value=1), nullable=False, coerce=True),
        "home_score": Column(int, Check.in_range(0, 99), nullable=False, coerce=True),
        "away_score": Column(int, Check.in_range(0, 99), nullable=False, coerce=True),
        "tournament": Column(str, Check.str_length(min_value=1), nullable=False, coerce=True),
        "city": Column(str, nullable=True, coerce=True),
        "country": Column(str, nullable=True, coerce=True),
        "neutral": Column(bool, nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
)


@dataclass(frozen=True)
class PipelineConfig:
    warehouse_path: Path = PROJECT_ROOT / "data" / "warehouse" / "wc26.duckdb"
    snapshot_root: Path = PROJECT_ROOT / "data" / "raw_snapshots"
    quarantine_root: Path = PROJECT_ROOT / "data" / "quarantine"
    artifacts_root: Path = PROJECT_ROOT / "data" / "published"
    dbt_project_dir: Path = PROJECT_ROOT / "warehouse"
    source_name: str = "international_results"
    schema_version: str = "1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_match_key(row: pd.Series | dict[str, Any]) -> str:
    """Return a score-independent key so corrected results upsert in place."""
    values = [
        str(pd.Timestamp(row["date"]).date()),
        normalize_team_name(str(row["home_team"])).casefold(),
        normalize_team_name(str(row["away_team"])).casefold(),
        str(row.get("tournament", "")).strip().casefold(),
        str(row.get("city", "")).strip().casefold(),
    ]
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def create_immutable_snapshot(source_path: Path, config: PipelineConfig) -> dict[str, Any]:
    """Copy a source file into a content-addressed immutable snapshot directory."""
    checksum = file_sha256(source_path)
    ingested_at = utc_now()
    snapshot_dir = config.snapshot_root / config.source_name / ingested_at.strftime("%Y/%m/%d")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{ingested_at.strftime('%H%M%S')}_{checksum[:12]}{source_path.suffix}"
    if snapshot_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable snapshot: {snapshot_path}")
    shutil.copy2(source_path, snapshot_path)
    record_count = int(len(pd.read_csv(source_path)))
    metadata = {
        "source_name": config.source_name,
        "source_path": str(source_path),
        "snapshot_path": str(snapshot_path),
        "ingestion_timestamp": ingested_at.isoformat(),
        "file_checksum": checksum,
        "schema_version": config.schema_version,
        "record_count": record_count,
    }
    snapshot_path.with_suffix(snapshot_path.suffix + ".manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def validate_and_quarantine(
    frame: pd.DataFrame,
    run_id: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate rows with Pandera and quarantine failures with a reason."""
    working = frame.copy()
    for column in RAW_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA
    working = working[RAW_COLUMNS]

    invalid_indexes: set[int] = set()
    reasons: dict[int, list[str]] = {}
    try:
        MATCH_SCHEMA.validate(working, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for _, failure in exc.failure_cases.iterrows():
            raw_index = failure.get("index")
            if pd.isna(raw_index):
                invalid_indexes.update(working.index.tolist())
                continue
            index = int(raw_index)
            invalid_indexes.add(index)
            reasons.setdefault(index, []).append(
                f"{failure.get('column', 'schema')}: {failure.get('check', 'invalid')}"
            )

    same_team = working["home_team"].astype(str).str.strip().eq(
        working["away_team"].astype(str).str.strip()
    )
    for index in working.index[same_team]:
        invalid_indexes.add(int(index))
        reasons.setdefault(int(index), []).append("home_team equals away_team")

    invalid = working.loc[sorted(invalid_indexes)].copy() if invalid_indexes else working.iloc[0:0].copy()
    if not invalid.empty:
        invalid["quarantine_reason"] = ["; ".join(reasons.get(int(i), ["schema validation failed"])) for i in invalid.index]
        invalid["run_id"] = run_id
        invalid["quarantined_at"] = utc_now().isoformat()
        config.quarantine_root.mkdir(parents=True, exist_ok=True)
        invalid.to_parquet(config.quarantine_root / f"{run_id}.parquet", index=False)

    valid = working.drop(index=sorted(invalid_indexes)).copy()
    valid = MATCH_SCHEMA.validate(valid)
    valid["source_home_team"] = valid["home_team"]
    valid["source_away_team"] = valid["away_team"]
    valid["source_tournament"] = valid["tournament"]
    valid["source_city"] = valid["city"]
    valid["source_country"] = valid["country"]
    valid["home_team"] = valid["home_team"].map(normalize_team_name)
    valid["away_team"] = valid["away_team"].map(normalize_team_name)
    valid["match_key"] = valid.apply(stable_match_key, axis=1)
    return valid, invalid


class Warehouse:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = duckdb.connect(str(path))
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            create table if not exists raw_matches (
                match_key varchar primary key,
                date date not null,
                home_team varchar not null,
                away_team varchar not null,
                home_score integer not null,
                away_score integer not null,
                tournament varchar not null,
                city varchar,
                country varchar,
                neutral boolean not null,
                source_home_team varchar,
                source_away_team varchar,
                source_tournament varchar,
                source_city varchar,
                source_country varchar,
                source_name varchar not null,
                snapshot_checksum varchar not null,
                ingested_at timestamp with time zone not null,
                updated_at timestamp with time zone not null
            );
            create table if not exists ingestion_checkpoints (
                source_name varchar primary key,
                watermark_date date,
                snapshot_checksum varchar,
                updated_at timestamp with time zone not null
            );
            create table if not exists pipeline_runs (
                run_id varchar primary key,
                started_at timestamp with time zone not null,
                finished_at timestamp with time zone,
                status varchar not null,
                source varchar not null,
                rows_extracted integer default 0,
                rows_loaded integer default 0,
                rows_rejected integer default 0,
                duplicates_removed integer default 0,
                models_built integer default 0,
                error_message varchar
            );
            create table if not exists quarantined_records (
                run_id varchar not null,
                match_key varchar,
                record_json varchar not null,
                reason varchar not null,
                quarantined_at timestamp with time zone not null
            );
            """
        )

    def start_run(self, run_id: str, source: str) -> None:
        self.connection.execute(
            "insert into pipeline_runs(run_id, started_at, status, source) values (?, ?, 'running', ?)",
            [run_id, utc_now(), source],
        )

    def finish_run(self, run_id: str, status: str, metrics: dict[str, Any], error: str | None = None) -> None:
        self.connection.execute(
            """
            update pipeline_runs set finished_at=?, status=?, rows_extracted=?, rows_loaded=?,
                rows_rejected=?, duplicates_removed=?, models_built=?, error_message=?
            where run_id=?
            """,
            [
                utc_now(), status, int(metrics.get("rows_extracted", 0)), int(metrics.get("rows_loaded", 0)),
                int(metrics.get("rows_rejected", 0)), int(metrics.get("duplicates_removed", 0)),
                int(metrics.get("models_built", 0)), error, run_id,
            ],
        )

    def watermark(self, source_name: str) -> pd.Timestamp | None:
        row = self.connection.execute(
            "select watermark_date from ingestion_checkpoints where source_name=?", [source_name]
        ).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    def load_matches(
        self,
        frame: pd.DataFrame,
        source_name: str,
        snapshot_checksum: str,
        backfill_start: str | None = None,
        backfill_end: str | None = None,
    ) -> dict[str, int]:
        incoming = frame.copy()
        if backfill_start:
            incoming = incoming[incoming["date"] >= pd.Timestamp(backfill_start)]
        if backfill_end:
            incoming = incoming[incoming["date"] <= pd.Timestamp(backfill_end)]
        if not backfill_start and not backfill_end:
            watermark = self.watermark(source_name)
            if watermark is not None:
                incoming = incoming[incoming["date"] >= watermark]

        before = len(incoming)
        incoming = incoming.sort_values(["date", "match_key"]).drop_duplicates("match_key", keep="last")
        duplicates_removed = before - len(incoming)
        if incoming.empty:
            return {"rows_loaded": 0, "duplicates_removed": duplicates_removed}

        incoming["source_name"] = source_name
        incoming["snapshot_checksum"] = snapshot_checksum
        incoming["ingested_at"] = utc_now()
        incoming["updated_at"] = utc_now()
        self.connection.register("incoming_matches", incoming)
        columns = [
            "match_key", "date", "home_team", "away_team", "home_score", "away_score", "tournament",
            "city", "country", "neutral", "source_home_team", "source_away_team", "source_tournament",
            "source_city", "source_country", "source_name", "snapshot_checksum", "ingested_at", "updated_at",
        ]
        names = ", ".join(columns)
        updates = ", ".join(f"{name}=excluded.{name}" for name in columns if name not in {"match_key", "ingested_at"})
        self.connection.execute(
            f"insert into raw_matches ({names}) select {names} from incoming_matches "
            f"on conflict(match_key) do update set {updates}"
        )
        max_date = incoming["date"].max()
        self.connection.execute(
            """
            insert into ingestion_checkpoints values (?, ?, ?, ?)
            on conflict(source_name) do update set watermark_date=excluded.watermark_date,
                snapshot_checksum=excluded.snapshot_checksum, updated_at=excluded.updated_at
            """,
            [source_name, max_date, snapshot_checksum, utc_now()],
        )
        return {"rows_loaded": int(len(incoming)), "duplicates_removed": int(duplicates_removed)}

    def load_quarantine(self, invalid: pd.DataFrame, run_id: str) -> None:
        for _, row in invalid.iterrows():
            self.connection.execute(
                "insert into quarantined_records values (?, ?, ?, ?, ?)",
                [
                    run_id,
                    row.get("match_key"),
                    json.dumps(row.drop(labels=["quarantine_reason"], errors="ignore").to_dict(), default=str),
                    row.get("quarantine_reason", "validation failed"),
                    utc_now(),
                ],
            )

    def load_feature_table(self, path: Path) -> int:
        table = pd.read_parquet(path)
        self.connection.register("feature_frame", table)
        self.connection.execute("create or replace table raw_model_features as select * from feature_frame")
        return int(len(table))


def _latest_results_path() -> Path:
    candidates = sorted((RAW_DIR / "international_results").rglob("results.csv"))
    if not candidates:
        raise FileNotFoundError("No international results source found")
    ranked = []
    for path in candidates:
        dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"], errors="coerce")
        ranked.append((dates.max(), path))
    return max(ranked, key=lambda item: item[0])[1]


def run_dbt(config: PipelineConfig, full_refresh: bool = False) -> None:
    dbt_executable = Path(sys.executable).with_name("dbt")
    if not dbt_executable.exists():
        located = shutil.which("dbt")
        if located is None:
            raise FileNotFoundError("dbt executable not found in the active Python environment")
        dbt_executable = Path(located)
    command = [
        str(dbt_executable), "build", "--project-dir", str(config.dbt_project_dir), "--profiles-dir",
        str(config.dbt_project_dir), "--target", "dev",
    ]
    if full_refresh:
        command.append("--full-refresh")
    environment = os.environ.copy()
    environment["WC26_WAREHOUSE_PATH"] = str(config.warehouse_path)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def publish_prediction_artifacts(config: PipelineConfig) -> dict[str, str]:
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    metadata_source = PROJECT_ROOT / "models" / "model_metadata.json"
    version = file_sha256(metadata_source)[:12]
    version_dir = config.artifacts_root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    metadata_target = version_dir / "model_metadata.json"
    shutil.copy2(metadata_source, metadata_target)
    latest = config.artifacts_root / "latest.json"
    latest.write_text(
        json.dumps({"model_version": version, "metadata_path": str(metadata_target), "published_at": utc_now().isoformat()}, indent=2),
        encoding="utf-8",
    )
    return {"model_version": version, "metadata_path": str(metadata_target), "latest_path": str(latest)}


def build_feature_base(source_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build features from complete raw inputs or use the documented incremental fallback.

    Elo and FIFA histories are intentionally not committed to the repository. A clean
    checkout therefore starts from the tracked, leakage-safe parquet and appends only
    scored rows newer than its watermark using the saved pre-match profiles.
    """
    try:
        frame, report = build_training_table(export_csv=False)
        return frame, {"mode": "full_raw_pipeline", "validation": report}
    except FileNotFoundError as exc:
        from src.automation.update_after_matchday import _incremental_fallback_training_table

        frame, fallback = _incremental_fallback_training_table(
            latest_results_path=source_path,
            export_csv=False,
        )
        return frame, {
            "mode": "incremental_profile_fallback",
            "reason": str(exc),
            "validation": fallback,
        }


def run_pipeline(
    *,
    config: PipelineConfig | None = None,
    backfill_start: str | None = None,
    backfill_end: str | None = None,
    rebuild_analytics: bool = False,
    train: bool = True,
) -> dict[str, Any]:
    """Execute snapshot, validation, upsert, dbt, training, and publishing."""
    cfg = config or PipelineConfig()
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    source_path = _latest_results_path()
    warehouse = Warehouse(cfg.warehouse_path)
    metrics: dict[str, Any] = {"run_id": run_id, "source": str(source_path)}
    warehouse.start_run(run_id, str(source_path))
    try:
        snapshot = create_immutable_snapshot(source_path, cfg)
        raw = pd.read_csv(source_path)
        metrics["rows_extracted"] = int(len(raw))
        valid, invalid = validate_and_quarantine(raw, run_id, cfg)
        metrics["rows_rejected"] = int(len(invalid))
        warehouse.load_quarantine(invalid, run_id)
        load_metrics = warehouse.load_matches(
            valid,
            cfg.source_name,
            snapshot["file_checksum"],
            backfill_start=backfill_start,
            backfill_end=backfill_end,
        )
        metrics.update(load_metrics)

        training_df, feature_build = build_feature_base(source_path)
        metrics["feature_rows"] = warehouse.load_feature_table(
            PROJECT_ROOT / "data" / "processed" / "training_matches.parquet"
        )
        run_dbt(cfg, full_refresh=rebuild_analytics)

        if train:
            model_metadata = train_baselines()
            metrics["models_built"] = 3
            metrics["model_metrics"] = model_metadata.get("metrics", {})
            metrics["model_split"] = model_metadata.get("split", {})
            metrics["published"] = publish_prediction_artifacts(cfg)
        else:
            metrics["models_built"] = 0

        metrics["feature_build"] = feature_build
        metrics["runtime_seconds"] = round(time.perf_counter() - started, 3)
        warehouse.finish_run(run_id, "success", metrics)
        return metrics
    except Exception as exc:
        metrics["runtime_seconds"] = round(time.perf_counter() - started, 3)
        warehouse.finish_run(run_id, "failed", metrics, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        warehouse.close()
