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

Refresh international results from zip first (recommended before rebuild):

```bash
python -m src.data.refresh_international_results --zip-path /Users/adeoluwa/Downloads/international_results-master-2.zip
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
- `data/config/wc26_teams.csv` with columns: `group, team, status, source, notes`.
- Simulator now uses this fixed group file for tournament input.
- Auto-generated Elo-seeded groups are disabled by default and only allowed when `allow_auto_groups_debug=True` and the fixed file is missing.

### Knockout Audit Note

Previous behavior used a generic seeded knockout pairing approach, which does not match the official 2026 structure and can skew title odds.

Current behavior uses:
- official Match 73-88 slot definitions,
- deterministic best-third routing via allowed-group combination lookup,
- fixed progression paths for Matches 89-102 plus third-place and final.

Diagnostic artifact:
- `data/processed/simulation_input_audit.json` (groups used, confirmed/projected slots, recent Elo sanity table, warnings).

Run Monte Carlo simulation:

```python
from src.simulation.monte_carlo import run_world_cup_simulation

result = run_world_cup_simulation(n_simulations=1000)
print(result.champion_probabilities.head())
```

Resolve projected WC26 qualifier slots in config from latest results/shootouts:

```bash
python -m src.simulation.update_wc26_teams --config-path data/config/wc26_teams.csv
```

## Streamlit Dashboard

Multi-page app includes:
- Overview
- Team Odds
- Group Winners
- Match Predictor
- Bracket

Run app:

```bash
streamlit run streamlit_app.py
```

### Team Photos In UI

To show country/team photos in predictor pages, add image files to:

`assets/team_photos/`

Supported extensions: `.avif`, `.png`, `.jpg`, `.jpeg`, `.webp`

Name files with team names (for example: `France.avif`, `Turkey.avif`). The app auto-loads images when available.

## Notes

- Evaluation is strictly time-based (no random split).
- Streamlit UI is modular and separate from simulation logic.
- Legacy Euro project is kept under `legacy/` for reference only.
