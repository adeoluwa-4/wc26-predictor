# Verification Report

Verified locally on 2026-10-24 with Python 3.12.13 and the repository's pinned runtime dependencies.

## End-to-end pipeline

Command: `wc26-pipeline run-all`

| Check | Result |
|---|---:|
| Source rows extracted | 49,547 |
| Rows rejected by schema validation | 0 |
| Rows present in `raw_matches` | 49,547 |
| Duplicate rows created | 0 |
| Rows in `analytics.mart_model_features` | 49,491 |
| dbt resources passed | 46 of 46 |
| Models trained | 3 |
| End-to-end runtime | 19.364 seconds |

The clean-checkout run used `incremental_profile_fallback` because the optional full raw Elo and FIFA histories are intentionally not tracked. The fallback rebuilt the warehouse from the tracked leakage-safe feature base and saved pre-match profiles; it did not fabricate historical ranking observations.

## Model evaluation

Model version: `20261024T171514Z-df35268f8fc3`

Data snapshot checksum: `df35268f8fc341ff7fb93d448b4e40356676ac35300a6b4461fd199a99ac1514`

Training source: `analytics.mart_model_features`

Chronological split: 4,660 training rows, 566 validation rows, and 237 test rows. The feature matrix contained 54 features.

| Target | Validation | Test |
|---|---:|---:|
| Match outcome accuracy | 63.25% | 63.29% |
| Match outcome log loss | 0.8127 | 0.8713 |
| Home goals MAE | 1.0436 | 1.0395 |
| Home goals RMSE | 1.3791 | 1.3625 |
| Away goals MAE | 0.9180 | 0.9313 |
| Away goals RMSE | 1.2509 | 1.2770 |

Training completed in 2.554 seconds and wrote SHA-256 checksums for every published model/profile artifact.

## Reliability checks

- `pytest`: 31 passed with no warnings after the Pandas string-dtype compatibility fix.
- `dbt build`: 11 models and 35 data tests completed successfully (`PASS=46`, `WARN=0`, `ERROR=0`).
- `dbt source freshness`: `warehouse.raw_matches` passed its freshness threshold.
- Idempotency replay: the warehouse remained at 49,547 matches after rerunning the same source. One watermark-date row was safely re-upserted and duplicate growth remained zero.
- Warehouse-to-model integration: a temporary DuckDB mart trained all model artifacts and verified their persisted SHA-256 checksums.
- Prefect: the local task graph completed ingest, validation, warehouse load, feature generation, and dbt successfully under run ID `bd6c7152-a6c2-450b-8bda-c8901c892884`. Training was disabled for that orchestration-only replay because the complete CLI run had already trained and published the measured artifacts above.
- Streamlit: the local server returned `HTTP/1.1 200 OK` and `ok` from `/_stcore/health`.

## Verification boundaries

- The GitHub Actions weekly schedule and manual backfill input are configured, but no hosted run was triggered during this local verification.
- The live Streamlit deployment was not redeployed or exercised in a browser. The result above is a local HTTP startup/health smoke test.
- DuckDB is the verified warehouse. No BigQuery or other cloud warehouse execution is claimed.
- The public repository cannot fully recompute historical Elo/FIFA joins until those optional source histories are supplied.
