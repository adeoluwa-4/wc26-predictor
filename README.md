# World Cup 2026 Predictor

An end-to-end machine-learning and simulation system for the 48-team FIFA World Cup. The project estimates match win, draw, and loss probabilities, predicts expected goals, and runs Monte Carlo tournament simulations to calculate advancement and title odds.

[Open the live Streamlit dashboard](https://adeoluwa-4-wc26-predictor-streamlit-app-awcr9s.streamlit.app)

## What the dashboard answers

- How likely is each team to win, draw, or lose a matchup?
- Which teams are most likely to advance from their groups?
- What is one team's probability of reaching each knockout round?
- Which teams are the strongest title contenders?
- How do played matches change the remaining tournament paths?

The dashboard includes Overview, Team Odds, Match Predictor, Group Winners, and Bracket views.

## System architecture

```mermaid
flowchart LR
    A["International results, Elo, and reference data"] --> B["Validation and team-name standardization"]
    B --> C["Rolling form, head-to-head, and tournament features"]
    C --> D["Chronological train / validation / test split"]
    D --> E["CatBoost outcome and goal models"]
    E --> F["Match probability interface"]
    F --> G["Monte Carlo tournament simulation"]
    G --> H["Streamlit dashboard"]
```

The data layer validates source schemas, normalizes historical team names, joins ratings, and builds time-aware rolling features. The model layer produces outcome probabilities and expected goals. The simulation layer applies official group and knockout rules across repeated tournament runs, while the Streamlit app turns the results into an interactive product.

## Features and modeling

The committed model uses 54 numeric and categorical features, including:

- Elo strength and strength difference
- Recent 5-match and 10-match form
- Goals scored, goals conceded, goal difference, and points per match
- Prior head-to-head record
- Confederation and same-confederation signals
- Tournament type and importance
- Host-country and neutral-venue context

Outcome prediction uses a multiclass CatBoost classifier. Separate CatBoost regressors estimate home and away goals. Team and head-to-head profiles are saved for consistent inference without recomputing the complete history for every dashboard request.

## Evaluation

The project uses a chronological split rather than a random split so evaluation only uses matches that occur after the training period. Friendlies are excluded from the current retained run.

| Target | Validation | Test |
| --- | ---: | ---: |
| Outcome accuracy | 62.50% | 61.81% |
| Outcome log loss | 0.819 | 0.892 |
| Home goals MAE | 1.074 | 0.996 |
| Away goals MAE | 0.920 | 0.867 |

These values come from `models/model_metadata.json`. They describe the committed evaluation split of 4,463 training matches, 720 validation matches, and 254 test matches; they are not claims of certainty for future matches.

## Repository structure

```text
src/data/          ingestion, validation, joins, and rolling features
src/models/        feature selection, training, tuning, and inference
src/simulation/    group tables, knockout rules, and Monte Carlo runs
src/app/           Streamlit dashboard, theming, flags, and images
src/automation/    refresh-and-retrain orchestration
data/config/       tournament teams and played-match state
models/            trained artifacts, profiles, and evaluation metadata
tests/             data, model-feature, and tournament-rule coverage
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the test suite with:

```bash
pytest
```

Retrain the committed baseline models with:

```bash
python -m src.models.train_baselines
```

## Automated refresh

`.github/workflows/update-after-matchday.yml` provides a manually dispatched GitHub Actions workflow that refreshes match data, retrains the models, and publishes changed data or model artifacts one file at a time. Played World Cup matches can be tracked in `data/config/wc26_played_matches.csv` so simulations respect completed fixtures.

## Limitations

- International football contains structural changes, sparse matchups, and events a historical model cannot anticipate.
- Probabilities depend on the quality and freshness of the underlying results and rating data.
- Monte Carlo estimates stabilize with more simulations but remain model-based estimates, not guarantees.
- Squad availability, injuries, and tactical changes are not fully represented unless reflected in the input data.
