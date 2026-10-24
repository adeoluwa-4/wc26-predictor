"""Prefect orchestration for the WC26 warehouse and ML workflow."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task

from src.models.train_baselines import train_baselines
from src.platform.pipeline import (
    PipelineConfig,
    Warehouse,
    _latest_results_path,
    build_feature_base,
    create_immutable_snapshot,
    publish_prediction_artifacts,
    run_dbt,
    validate_and_quarantine,
)
from src.utils.paths import PROJECT_ROOT


@task(retries=2, retry_delay_seconds=10, task_run_name="ingest-{run_id}")
def ingest(run_id: str, config: PipelineConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    logger = get_run_logger()
    source = _latest_results_path()
    manifest = create_immutable_snapshot(source, config)
    frame = pd.read_csv(source)
    logger.info("run_id=%s stage=ingest source=%s rows=%d", run_id, source, len(frame))
    return manifest, frame


@task(retries=1, retry_delay_seconds=5, task_run_name="validate-{run_id}")
def validate(run_id: str, config: PipelineConfig, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()
    valid, invalid = validate_and_quarantine(frame, run_id, config)
    logger.info("run_id=%s stage=validate valid=%d rejected=%d", run_id, len(valid), len(invalid))
    return valid, invalid


@task(retries=2, retry_delay_seconds=10, task_run_name="load-{run_id}")
def load_warehouse(
    run_id: str,
    config: PipelineConfig,
    manifest: dict[str, Any],
    valid: pd.DataFrame,
    invalid: pd.DataFrame,
    backfill_start: str | None,
    backfill_end: str | None,
) -> dict[str, int]:
    logger = get_run_logger()
    warehouse = Warehouse(config.warehouse_path)
    try:
        warehouse.load_quarantine(invalid, run_id)
        result = warehouse.load_matches(
            valid,
            config.source_name,
            manifest["file_checksum"],
            backfill_start,
            backfill_end,
        )
    finally:
        warehouse.close()
    logger.info("run_id=%s stage=load metrics=%s", run_id, result)
    return result


@task(retries=1, retry_delay_seconds=5, task_run_name="features-{run_id}")
def generate_features(run_id: str, config: PipelineConfig, source_path: str) -> dict[str, Any]:
    logger = get_run_logger()
    frame, report = build_feature_base(Path(source_path))
    warehouse = Warehouse(config.warehouse_path)
    try:
        rows = warehouse.load_feature_table(PROJECT_ROOT / "data" / "processed" / "training_matches.parquet")
    finally:
        warehouse.close()
    logger.info("run_id=%s stage=features rows=%d mode=%s", run_id, rows, report["mode"])
    return {"feature_rows": rows, "feature_build": report}


@task(retries=1, retry_delay_seconds=5, task_run_name="dbt-{run_id}")
def build_analytics(run_id: str, config: PipelineConfig, full_refresh: bool) -> None:
    run_dbt(config, full_refresh=full_refresh)
    get_run_logger().info("run_id=%s stage=dbt status=success", run_id)


@task(retries=1, retry_delay_seconds=5, task_run_name="train-{run_id}")
def train_and_evaluate(run_id: str) -> dict[str, Any]:
    metadata = train_baselines()
    get_run_logger().info("run_id=%s stage=train metrics=%s", run_id, metadata.get("metrics", {}))
    return metadata


@task(retries=1, retry_delay_seconds=5, task_run_name="publish-{run_id}")
def publish(run_id: str, config: PipelineConfig) -> dict[str, str]:
    artifact = publish_prediction_artifacts(config)
    get_run_logger().info("run_id=%s stage=publish model_version=%s", run_id, artifact["model_version"])
    return artifact


@flow(name="wc26-data-platform", log_prints=True)
def wc26_pipeline_flow(
    backfill_start: str | None = None,
    backfill_end: str | None = None,
    full_refresh: bool = False,
    train_models: bool = True,
) -> dict[str, Any]:
    """Run the explicit ingest-to-publish workflow with persisted run status."""
    config = PipelineConfig()
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    warehouse = Warehouse(config.warehouse_path)
    source = str(_latest_results_path())
    warehouse.start_run(run_id, source)
    warehouse.close()
    summary: dict[str, Any] = {"run_id": run_id, "source": source}

    try:
        manifest, raw = ingest(run_id, config)
        valid, invalid = validate(run_id, config, raw)
        summary["rows_extracted"] = len(raw)
        summary["rows_rejected"] = len(invalid)
        summary.update(
            load_warehouse(
                run_id,
                config,
                manifest,
                valid,
                invalid,
                backfill_start,
                backfill_end,
            )
        )
        summary.update(generate_features(run_id, config, source))
        build_analytics(run_id, config, full_refresh)
        if train_models:
            metadata = train_and_evaluate(run_id)
            summary["models_built"] = 3
            summary["model_metrics"] = metadata.get("metrics", {})
            summary["model_split"] = metadata.get("split", {})
            summary["published"] = publish(run_id, config)
        else:
            summary["models_built"] = 0
        summary["runtime_seconds"] = round(time.perf_counter() - started, 3)
        status = "success"
        error = None
    except Exception as exc:
        summary["runtime_seconds"] = round(time.perf_counter() - started, 3)
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        warehouse = Warehouse(config.warehouse_path)
        try:
            warehouse.finish_run(run_id, status, summary, error)
        finally:
            warehouse.close()

    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    wc26_pipeline_flow()
