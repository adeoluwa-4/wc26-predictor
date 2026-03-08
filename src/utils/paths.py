"""Path helpers for WC26 Predictor."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_directories() -> None:
    """Create expected data directories if they do not already exist."""
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    """Build an absolute path from the project root."""
    return PROJECT_ROOT.joinpath(*parts)
