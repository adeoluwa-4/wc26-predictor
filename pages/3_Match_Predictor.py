"""Direct match prediction page using trained inference interface."""

from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from src.app.dashboard import get_team_options, render_sidebar
from src.models.predict_interface import predict_match


st.set_page_config(page_title="Match Predictor | WC26 Predictor", layout="wide")

state = render_sidebar(default_team="United States")

st.title("Match Predictor")
st.caption("Predict a single matchup with win/draw/loss probabilities and expected scoreline.")

teams = get_team_options()
default_home_idx = teams.index(state.selected_team) if state.selected_team in teams else 0
home_team = st.selectbox("Home Team", teams, index=default_home_idx)
away_candidates = [t for t in teams if t != home_team]
away_team = st.selectbox("Away Team", away_candidates, index=0)

if st.button("Predict Match", type="primary"):
    out = predict_match(home_team=home_team, away_team=away_team)

    home_prob = out["home_win_probability"] * 100.0
    draw_prob = out["draw_probability"] * 100.0
    away_prob = out["away_win_probability"] * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} Win", f"{home_prob:.2f}%")
    c2.metric("Draw", f"{draw_prob:.2f}%")
    c3.metric(f"{away_team} Win", f"{away_prob:.2f}%")

    prob_df = pd.DataFrame(
        {
            "Outcome": [f"{home_team} Win", "Draw", f"{away_team} Win"],
            "Probability": [home_prob, draw_prob, away_prob],
        }
    )

    fig = px.bar(
        prob_df,
        x="Outcome",
        y="Probability",
        title="Outcome Probabilities",
        labels={"Probability": "Probability (%)"},
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    predicted_home_goals = out["predicted_home_goals"]
    predicted_away_goals = out["predicted_away_goals"]
    confidence = max(home_prob, draw_prob, away_prob)

    st.subheader("Score and Confidence")
    st.write(f"Predicted scoreline (expected goals): **{home_team} {predicted_home_goals:.2f} - {predicted_away_goals:.2f} {away_team}**")
    st.write(f"Model confidence (max class probability): **{confidence:.2f}%**")

    st.dataframe(
        pd.DataFrame(
            {
                "Metric": ["Expected Home Goals", "Expected Away Goals", "Confidence"],
                "Value": [f"{predicted_home_goals:.2f}", f"{predicted_away_goals:.2f}", f"{confidence:.2f}%"],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
