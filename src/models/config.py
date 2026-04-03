"""Configuration for baseline modeling stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.paths import PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "training_matches.parquet"


@dataclass(frozen=True)
class ModelingConfig:
    train_frac: float = 0.80
    val_frac: float = 0.10
    random_state: int = 42
    min_training_date: str | None = "2019-01-01"
    exclude_friendlies: bool = True
    outcome_model: str = "catboost"
    goal_model: str = "catboost"
    drop_categorical_features: list[str] = field(default_factory=list)
    catboost_depth: int = 3
    catboost_learning_rate: float = 0.08
    catboost_iterations: int = 300
    catboost_l2_leaf_reg: int = 2
    catboost_goal_depth: int = 4
    catboost_goal_learning_rate: float = 0.05
    catboost_goal_iterations: int = 400
