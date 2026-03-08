"""Configuration for baseline modeling stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "training_matches.parquet"


@dataclass(frozen=True)
class ModelingConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    random_state: int = 42
