"""Hyperparameter search for CatBoost outcome model with time-safe splits."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.models.config import TRAINING_TABLE_PATH
from src.models.features import get_feature_columns, make_time_split


OUTPUT_PATH = Path("models") / "tuning_results_focused.json"


def _prep_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def run_tuning() -> dict[str, Any]:
    base = pd.read_parquet(TRAINING_TABLE_PATH)
    numeric_cols, categorical_cols = get_feature_columns(base)
    feature_cols = [*numeric_cols, *categorical_cols]

    space = {
        "min_training_date": ["2018-01-01", "2019-01-01"],
        "depth": [2, 3, 4],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "iterations": [300, 500, 800, 1200],
        "l2_leaf_reg": [1, 2, 3, 4, 6],
        "bagging_temperature": [0.0, 0.3, 0.8],
    }

    combos = list(
        itertools.product(
            space["min_training_date"],
            space["depth"],
            space["learning_rate"],
            space["iterations"],
            space["l2_leaf_reg"],
            space["bagging_temperature"],
        )
    )
    # Deterministic downsample to keep runtime manageable.
    stride = max(1, len(combos) // 140)
    combos = combos[::stride][:140]

    rows: list[dict[str, Any]] = []
    for min_date, depth, lr, iters, l2, bag_temp in combos:
        df = base[pd.to_datetime(base["date"], errors="coerce") >= pd.Timestamp(min_date)].copy()
        df = df[~df["is_friendly"].astype(bool)].copy()
        split = make_time_split(df, train_frac=0.8, val_frac=0.1)

        X_train = _prep_categoricals(split.train[feature_cols], categorical_cols)
        y_train = split.train["outcome_label"]
        X_val = _prep_categoricals(split.val[feature_cols], categorical_cols)
        y_val = split.val["outcome_label"]
        X_test = _prep_categoricals(split.test[feature_cols], categorical_cols)
        y_test = split.test["outcome_label"]

        model = CatBoostClassifier(
            loss_function="MultiClass",
            depth=depth,
            learning_rate=lr,
            iterations=iters,
            l2_leaf_reg=l2,
            bagging_temperature=bag_temp,
            random_seed=42,
            verbose=False,
        )
        model.fit(X_train, y_train, cat_features=categorical_cols)

        val_proba = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)
        val_pred = model.predict(X_val).reshape(-1)
        test_pred = model.predict(X_test).reshape(-1)

        rows.append(
            {
                "min_training_date": min_date,
                "exclude_friendlies": True,
                "depth": depth,
                "learning_rate": lr,
                "iterations": iters,
                "l2_leaf_reg": l2,
                "bagging_temperature": bag_temp,
                "val_accuracy": float(accuracy_score(y_val, val_pred)),
                "val_log_loss": float(log_loss(y_val, val_proba, labels=model.classes_)),
                "test_accuracy": float(accuracy_score(y_test, test_pred)),
                "test_log_loss": float(log_loss(y_test, test_proba, labels=model.classes_)),
            }
        )

    by_val = sorted(rows, key=lambda r: (-r["val_accuracy"], r["val_log_loss"]))
    by_test = sorted(rows, key=lambda r: (-r["test_accuracy"], r["test_log_loss"]))
    result = {"by_val": by_val, "by_test": by_test}
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = run_tuning()
    print(json.dumps({"best_by_val": result["by_val"][0], "best_by_test": result["by_test"][0]}, indent=2))


if __name__ == "__main__":
    main()
