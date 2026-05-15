"""Refresh raw international results files from a zip archive."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.logging import configure_logging, get_logger
from src.utils.paths import RAW_DIR

LOGGER = get_logger(__name__)


REQUIRED_FILES = ("results.csv", "shootouts.csv", "former_names.csv")


def _find_zip_member(zf: zipfile.ZipFile, filename: str) -> str:
    members = [name for name in zf.namelist() if name.endswith(filename) and "__MACOSX" not in name]
    if not members:
        raise FileNotFoundError(f"{filename} not found in archive")
    members.sort(key=len)
    return members[0]


def _infer_bundle_name(results_member: str, zip_path: Path) -> str:
    parts = Path(results_member).parts
    if len(parts) > 1:
        return parts[0]
    return zip_path.stem


def refresh_from_zip(zip_path: Path, output_root: Path) -> dict[str, object]:
    """Extract key raw files from zip into data/raw/international_results/<bundle>/."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        members = {filename: _find_zip_member(zf, filename) for filename in REQUIRED_FILES}
        bundle_name = _infer_bundle_name(members["results.csv"], zip_path)
        output_dir = output_root / bundle_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for filename, member in members.items():
            (output_dir / filename).write_bytes(zf.read(member))

    results_df = pd.read_csv(output_dir / "results.csv", usecols=["date"])
    results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
    return {
        "output_dir": str(output_dir),
        "rows": int(len(results_df)),
        "min_date": str(results_df["date"].min().date()),
        "max_date": str(results_df["date"].max().date()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh raw international results from a zip file.")
    parser.add_argument(
        "--zip-path",
        default="/Users/adeoluwa/Downloads/international_results-master-2.zip",
        help="Path to the zip archive containing results.csv/shootouts.csv/former_names.csv",
    )
    parser.add_argument(
        "--output-root",
        default=str(RAW_DIR / "international_results"),
        help="Directory where extracted bundle folder should be written",
    )
    args = parser.parse_args()

    configure_logging()
    summary = refresh_from_zip(Path(args.zip_path), Path(args.output_root))
    LOGGER.info(
        "Refreshed international results: output_dir=%s rows=%s min_date=%s max_date=%s",
        summary["output_dir"],
        summary["rows"],
        summary["min_date"],
        summary["max_date"],
    )


if __name__ == "__main__":
    main()
