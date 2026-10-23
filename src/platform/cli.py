"""Command-line interface for repeatable WC26 data-platform operations."""

from __future__ import annotations

import argparse
import json

from src.models.train_baselines import train_baselines
from src.platform.pipeline import PipelineConfig, run_dbt, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wc26-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("refresh", help="Run an incremental refresh from the latest source snapshot")

    backfill = commands.add_parser("backfill", help="Upsert a historical date range")
    backfill.add_argument("--start", required=True, help="Inclusive YYYY-MM-DD start")
    backfill.add_argument("--end", required=True, help="Inclusive YYYY-MM-DD end")

    commands.add_parser("rebuild", help="Rebuild all dbt analytics models")
    commands.add_parser("train", help="Train CatBoost models from mart_model_features")
    commands.add_parser("run-all", help="Run the complete local pipeline")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = PipelineConfig()

    if args.command == "refresh":
        result = run_pipeline(config=config, train=False)
    elif args.command == "backfill":
        result = run_pipeline(
            config=config,
            backfill_start=args.start,
            backfill_end=args.end,
            train=False,
        )
    elif args.command == "rebuild":
        run_dbt(config, full_refresh=True)
        result = {"status": "success", "warehouse": str(config.warehouse_path)}
    elif args.command == "train":
        metadata = train_baselines()
        result = {"status": "success", "metrics": metadata.get("metrics", {})}
    else:
        result = run_pipeline(config=config, rebuild_analytics=True, train=True)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
