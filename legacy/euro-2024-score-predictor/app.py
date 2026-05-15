import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# -------------------------------
# ✅ Must be first Streamlit call
st.set_page_config(page_title="Euro 2024 Predictor", layout="centered")
# -------------------------------

# 📷 Load and show logo/image
image = Image.open("assets/euro-2024.jpeg")
st.image(image, use_column_width=True)

# 🔢 Load models
home_xg_model = joblib.load("models/home_xg_model.pkl")
away_xg_model = joblib.load("models/away_xg_model.pkl")
outcome_model = joblib.load("models/outcome_model.pkl")

#  Title
st.title("Euro 2024 Match Predictor")
st.markdown("Select two teams and we’ll predict the score & result based on historical stats.")

#  Team options based on model training
teams = sorted(list(set(
    [col.replace("home_team_", "") for col in home_xg_model.feature_names_in_ if col.startswith("home_team_")]
)))

# Team inputs
home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", [team for team in teams if team != home_team])

if st.button("Predict Result"):
    #  Create input DataFrame
    input_df = pd.DataFrame([{
        "home_team": home_team,
        "away_team": away_team
    }])

    # One-hot encode
    encoded_input = pd.get_dummies(input_df)

    # Align with training features
    encoded_input = encoded_input.reindex(columns=home_xg_model.feature_names_in_, fill_value=0)

    #  Predict xG
    home_xg = home_xg_model.predict(encoded_input)[0]
    away_xg = away_xg_model.predict(encoded_input)[0]

    #  Prepare outcome prediction
    outcome_input = pd.DataFrame([{
        "Home Expected goals(xG)": home_xg,
        "Away Expected goals(xG)": away_xg
    }])

    #  Align outcome features
    outcome_input = outcome_input.reindex(columns=outcome_model.feature_names_in_, fill_value=0)

    # 🎯 Predict match outcome
    result = outcome_model.predict(outcome_input)[0]

    #  Translate result
    result_text = {
        1: f"{home_team} Win",
        2: f"{away_team} Win",
        0: "Draw"
    }

    # 📊 Output
    st.subheader("Predicted Score:")
    st.markdown(f"**{home_team} xG:** {home_xg:.2f}")
    st.markdown(f"**{away_team} xG:** {away_xg:.2f}")
    st.success(f"**Predicted Outcome:** {result_text[result]}")
