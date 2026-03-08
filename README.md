# World Cup 2026 Predictor 
This repository contains:
- a reproducible, date safe data pipeline,
- baseline model training with time based evaluation,
- and an inference interface for match predictions.

## Data Pipeline Artifacts

Pipeline outputs:
- `data/processed/training_matches.parquet`
- `data/processed/training_matches.csv` (optional)
- `data/processed/validation_report.json`
- `data/processed/unmapped_team_names.csv`

Build pipeline:

```bash
python -m src.data.build_training_table --export-csv
```

## Baseline Modeling Stage

Trains and saves:
- `models/outcome_model.joblib`
- `models/home_goals_model.joblib`
- `models/away_goals_model.joblib`
- `models/model_metadata.json`
- `models/team_profiles.parquet`
- `models/h2h_profiles.parquet`

Run training:

```bash
python -m src.models.train_baselines
```

## Inference Interface

Use from Python:

```python
from src.models.predict_interface import predict_match

out = predict_match("Brazil", "France")
print(out)
```

Returned keys:
- `home_win_probability`
- `draw_probability`
- `away_win_probability`
- `predicted_home_goals`
- `predicted_away_goals`

## Tournament Simulation Engine

Simulation modules live in `src/simulation/` and support:
- group stage simulation,
- third-place ranking,
- knockout bracket progression,
- Monte Carlo aggregation.

Teams are loaded from:
- `data/config/wc26_teams.csv` with columns: `team`, `confederation`, `pot`, optional `group`.
- If `group` is blank, groups are drawn randomly with confederation constraints (max 2 UEFA, max 1 from other confederations, and one team per pot per group).

Run Monte Carlo simulation:

```python
from src.simulation.monte_carlo import run_world_cup_simulation

result = run_world_cup_simulation(n_simulations=1000)
print(result.champion_probabilities.head())
```

## Streamlit Dashboard

Multi-page app includes:
- Overview
- Team Odds
- Group Winners
- Match Predictor

Run app:

```bash
streamlit run streamlit_app.py
```

## Notes

- Evaluation is strictly time-based (no random split).
- Streamlit UI is modular and separate from simulation logic.
- Legacy Euro project is kept under `legacy/` for reference only.
