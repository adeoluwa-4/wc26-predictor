# Data Dictionary and Lineage

## Raw operational tables

### `main.raw_matches`

| Column | Meaning |
|---|---|
| `match_key` | SHA-256 key built from date, canonical teams, competition, city, and occurrence; scores are excluded so corrections upsert |
| `date` | Match date |
| `home_team`, `away_team` | Normalized team names |
| `home_score`, `away_score` | Final regulation/recorded scores |
| `tournament`, `city`, `country`, `neutral` | Match context |
| `match_occurrence` | Zero-based discriminator for genuine repeat fixtures with otherwise identical identity fields |
| `source_*` | Original source values retained for traceability |
| `source_name` | Ingestion source identifier |
| `snapshot_checksum` | SHA-256 checksum of the immutable source snapshot |
| `ingested_at`, `updated_at` | Initial ingestion and latest upsert timestamps |

### `main.ingestion_checkpoints`

One row per source with `watermark_date`, the latest `snapshot_checksum`, and `updated_at`. Incremental loads reread the watermark date to support late same-day records.

### `main.pipeline_runs`

| Column | Meaning |
|---|---|
| `run_id` | UUID shared across orchestration logs and outputs |
| `started_at`, `finished_at` | UTC run boundaries |
| `status` | `running`, `success`, or `failed` |
| `source` | Source file used by the run |
| `rows_extracted`, `rows_loaded`, `rows_rejected` | Ingestion counts |
| `duplicates_removed` | Duplicate stable keys removed from the incoming batch |
| `models_built` | Number of model artifacts trained |
| `error_message` | Terminal failure detail when present |

### `main.quarantined_records`

Invalid row payloads keyed by `run_id`, with a validation `reason` and `quarantined_at` timestamp. A parquet copy is also written under `data/quarantine/`.

### `main.raw_model_features`

The leakage-safe Python feature output registered in DuckDB before dbt builds the model contract. This separation keeps existing feature logic and tests intact while making its output queryable and governed.

## dbt lineage

```text
main.raw_matches
  -> analytics.stg_matches
      -> analytics.int_team_match_history
          -> analytics.dim_team
          -> analytics.mart_team_performance
      -> analytics.dim_tournament
      -> analytics.dim_date
      -> analytics.fact_match

main.raw_model_features + analytics.stg_matches
  -> analytics.fact_team_ranking
  -> analytics.fact_team_form

main.raw_model_features
  -> analytics.mart_model_features
      -> CatBoost outcome/home-goals/away-goals artifacts

main.pipeline_runs
  -> analytics.mart_predictions
```

## Analytics models

### `analytics.dim_team`

One row per canonical team with `team_key`, `team_name`, `first_match_date`, and `latest_match_date`.

### `analytics.dim_tournament`

One row per competition with a deterministic key, normalized type, match count, and observed date range.

### `analytics.dim_date`

One row per match date with integer `date_key`, year, quarter, month, day, weekday, and ISO-style week number.

### `analytics.fact_match`

One row per stable match key. Foreign keys reference the date, home team, away team, and tournament dimensions. Measures include scores, total goals, goal difference, and outcome.

### `analytics.fact_team_ranking`

Two rows per linked match, one per team side, containing the strictly prior-date Elo, FIFA rank, and FIFA points used for that match.

### `analytics.fact_team_form`

Two rows per linked match, one per team side, containing prior-only 5-match and 10-match points, points per match, goal difference, and wins.

### `analytics.mart_team_performance`

Team-level totals and rates for dashboard analytics: matches, wins/draws/losses, points, goals, goal difference, points per match, scoring/conceding rates, and date range.

### `analytics.mart_model_features`

The CatBoost training contract. It preserves the existing 54 selected features, labels, match metadata, chronological feature construction, and adds a `feature_row_checksum` for lineage.

### `analytics.mart_predictions`

Successful pipeline/model publication runs with version, publication time, source, row-quality counts, model count, and status.
