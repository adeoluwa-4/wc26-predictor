import pandas as pd
import joblib

# Load model
model = joblib.load("models/euro2024_rf_model.pkl")

# Function to make prediction
def predict_outcome(home_team, away_team, home_xg, away_xg):
    # Create DataFrame with all possible teams from training
    all_teams = [col for col in model.feature_names_in_ if "home_team_" in col or "away_team_" in col]
    input_data = {col: 0 for col in all_teams}
    
    # Set correct one-hot values
    input_data[f"home_team_{home_team}"] = 1
    input_data[f"away_team_{away_team}"] = 1

    # Add xG values
    input_data["Home Expected goals(xG)"] = home_xg
    input_data["Away Expected goals(xG)"] = away_xg

    # Convert to DataFrame
    X = pd.DataFrame([input_data])

    # Predict
    outcome = model.predict(X)[0]

    if outcome == 1:
        return f"{home_team} wins"
    elif outcome == 2:
        return f"{away_team} wins"
    else:
        return "Draw"
