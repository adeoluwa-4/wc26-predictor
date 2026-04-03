"""Train baseline outcome and goal models with time-based evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.config import MODELS_DIR, TRAINING_TABLE_PATH, ModelingConfig
from src.models.features import get_feature_columns, make_time_split


OUTCOME_MODEL_PATH = MODELS_DIR / "outcome_model.joblib"
HOME_GOALS_MODEL_PATH = MODELS_DIR / "home_goals_model.joblib"
AWAY_GOALS_MODEL_PATH = MODELS_DIR / "away_goals_model.joblib"
TEAM_PROFILE_PATH = MODELS_DIR / "team_profiles.parquet"
H2H_PROFILE_PATH = MODELS_DIR / "h2h_profiles.parquet"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def _fit_outcome_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: ModelingConfig,
) -> tuple[Any, dict[str, float], dict[str, float], list[str]]:
    X_train = train_df[feature_cols]
    y_train = train_df["outcome_label"]

    if config.outcome_model == "catboost":
        model = CatBoostClassifier(
            loss_function="MultiClass",
            depth=config.catboost_depth,
            learning_rate=config.catboost_learning_rate,
            iterations=config.catboost_iterations,
            l2_leaf_reg=config.catboost_l2_leaf_reg,
            random_seed=config.random_state,
            verbose=False,
        )
        X_train_model = X_train.copy()
        for col in categorical_cols:
            X_train_model[col] = X_train_model[col].fillna("Unknown").astype(str)
        model.fit(X_train_model, y_train, cat_features=categorical_cols)

        def evaluate(split_df: pd.DataFrame) -> dict[str, float]:
            X = split_df[feature_cols].copy()
            for col in categorical_cols:
                X[col] = X[col].fillna("Unknown").astype(str)
            y = split_df["outcome_label"]
            pred = model.predict(X).reshape(-1)
            proba = model.predict_proba(X)
            classes = model.classes_
            return {
                "accuracy": float(accuracy_score(y, pred)),
                "log_loss": float(log_loss(y, proba, labels=classes)),
            }

        return model, evaluate(val_df), evaluate(test_df), list(model.classes_)

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    logistic = LogisticRegression(
        max_iter=20000,
        solver="saga",
        n_jobs=1,
        tol=5e-4,
        C=4.0,
        class_weight=None,
        random_state=config.random_state,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", logistic),
        ]
    )
    model.fit(X_train, y_train)

    def evaluate(split_df: pd.DataFrame) -> dict[str, float]:
        X = split_df[feature_cols]
        y = split_df["outcome_label"]
        pred = model.predict(X)
        proba = model.predict_proba(X)
        classes = model.named_steps["model"].classes_
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "log_loss": float(log_loss(y, proba, labels=classes)),
        }

    return model, evaluate(val_df), evaluate(test_df), list(model.named_steps["model"].classes_)


def _fit_goal_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[Pipeline, dict[str, float], dict[str, float]]:
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    model = PoissonRegressor(alpha=0.2, max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    pipeline.fit(X_train, y_train)

    def evaluate(split_df: pd.DataFrame) -> dict[str, float]:
        X = split_df[feature_cols]
        y = split_df[target_col]
        pred = pipeline.predict(X)
        rmse = mean_squared_error(y, pred) ** 0.5
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(rmse),
        }

    return pipeline, evaluate(val_df), evaluate(test_df)


def _build_team_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create latest per-team feature profile for inference without explicit match date input."""
    rolling_base = sorted(
        {
            col.replace("home_", "")
            for col in df.columns
            if col.startswith("home_") and col.endswith(("_last_5", "_last_10"))
        }
    )

    def side_frame(side: str) -> pd.DataFrame:
        team_col = f"{side}_team"
        cols = [
            "date",
            team_col,
            f"{side}_elo",
            f"{side}_fifa_rank",
            f"{side}_fifa_points",
            f"confederation_{side}",
        ] + [f"{side}_{metric}" for metric in rolling_base]

        side_df = df[cols].copy()
        rename_map = {
            team_col: "team",
            f"{side}_elo": "elo",
            f"{side}_fifa_rank": "fifa_rank",
            f"{side}_fifa_points": "fifa_points",
            f"confederation_{side}": "confederation",
        }
        rename_map.update({f"{side}_{metric}": metric for metric in rolling_base})
        return side_df.rename(columns=rename_map)

    combined = pd.concat([side_frame("home"), side_frame("away")], ignore_index=True)
    combined = combined.sort_values(["team", "date"]).drop_duplicates(subset=["team"], keep="last")
    combined = combined.reset_index(drop=True)

    numeric_profile_cols = [c for c in combined.columns if c not in {"team", "date", "confederation"}]
    defaults = {
        "confederation": (
            combined["confederation"].dropna().mode().iloc[0] if not combined["confederation"].dropna().empty else "Unknown"
        ),
        "numeric": {col: float(combined[col].median(skipna=True)) for col in numeric_profile_cols},
    }

    return combined, defaults


def _build_h2h_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        team_a, team_b = sorted((home, away))
        score_a = home_score if home == team_a else away_score
        score_b = away_score if home == team_a else home_score

        rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "team_a_wins": int(score_a > score_b),
                "team_b_wins": int(score_b > score_a),
                "draws": int(score_a == score_b),
                "team_a_goal_diff": int(score_a - score_b),
            }
        )

    h2h = pd.DataFrame(rows)
    summary = (
        h2h.groupby(["team_a", "team_b"], as_index=False)
        .agg(
            matches=("team_a_wins", "count"),
            team_a_wins=("team_a_wins", "sum"),
            team_b_wins=("team_b_wins", "sum"),
            draws=("draws", "sum"),
            team_a_goal_diff=("team_a_goal_diff", "sum"),
        )
        .sort_values(["team_a", "team_b"])
        .reset_index(drop=True)
    )
    return summary


def train_baselines(config: ModelingConfig | None = None) -> dict[str, Any]:
    config = config or ModelingConfig()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TRAINING_TABLE_PATH)
    if config.min_training_date:
        cutoff = pd.Timestamp(config.min_training_date)
        df = df[pd.to_datetime(df["date"], errors="coerce") >= cutoff].copy()
        if df.empty:
            raise ValueError(f"No rows remain after min_training_date={config.min_training_date}")
    if config.exclude_friendlies:
        df = df[~df["is_friendly"].astype(bool)].copy()
        if df.empty:
            raise ValueError("No rows remain after exclude_friendlies filter")

    split = make_time_split(df, train_frac=config.train_frac, val_frac=config.val_frac)

    numeric_cols, categorical_cols = get_feature_columns(df)
    if config.drop_categorical_features:
        categorical_cols = [col for col in categorical_cols if col not in set(config.drop_categorical_features)]
    feature_cols = [*numeric_cols, *categorical_cols]

    outcome_model, outcome_val, outcome_test, outcome_classes = _fit_outcome_model(
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        config=config,
    )

    home_goal_model, home_goal_val, home_goal_test = _fit_goal_model(
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        target_col="home_score",
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    away_goal_model, away_goal_val, away_goal_test = _fit_goal_model(
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        target_col="away_score",
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    team_profiles, defaults = _build_team_profiles(df)
    h2h_profiles = _build_h2h_profiles(df)

    joblib.dump(outcome_model, OUTCOME_MODEL_PATH)
    joblib.dump(home_goal_model, HOME_GOALS_MODEL_PATH)
    joblib.dump(away_goal_model, AWAY_GOALS_MODEL_PATH)
    team_profiles.to_parquet(TEAM_PROFILE_PATH, index=False)
    h2h_profiles.to_parquet(H2H_PROFILE_PATH, index=False)

    metadata = {
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "split": {
            "min_training_date": config.min_training_date,
            "exclude_friendlies": config.exclude_friendlies,
            "train_end_date": str(split.train_end_date.date()),
            "val_end_date": str(split.val_end_date.date()),
            "train_rows": int(len(split.train)),
            "val_rows": int(len(split.val)),
            "test_rows": int(len(split.test)),
        },
        "defaults": defaults,
        "metrics": {
            "outcome": {"val": outcome_val, "test": outcome_test},
            "home_goals": {"val": home_goal_val, "test": home_goal_test},
            "away_goals": {"val": away_goal_val, "test": away_goal_test},
        },
        "outcome_model": config.outcome_model,
        "outcome_model_params": {
            "catboost_depth": config.catboost_depth,
            "catboost_learning_rate": config.catboost_learning_rate,
            "catboost_iterations": config.catboost_iterations,
            "catboost_l2_leaf_reg": config.catboost_l2_leaf_reg,
        }
        if config.outcome_model == "catboost"
        else {},
        "catboost_categorical_columns": categorical_cols if config.outcome_model == "catboost" else [],
        "outcome_classes": outcome_classes,
    }

    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = train_baselines()
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
