"""Direct match prediction page using trained inference interface."""

from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from src.app.dashboard import get_team_options, render_sidebar
from src.app.team_images import team_photo_path
from src.app.team_flags import team_with_flag
from src.app.theme import apply_wc26_theme
from src.models.predict_interface import predict_match


st.set_page_config(page_title="Match Predictor | World Cup 2026 Predictor", layout="wide")
apply_wc26_theme()

state = render_sidebar(default_team="United States")

st.title("Match Predictor")
st.caption("Predict a single matchup with win/draw/loss probabilities and expected scoreline.")

teams = get_team_options()
default_home_idx = teams.index(state.selected_team) if state.selected_team in teams else 0
home_team = st.selectbox("Home Team", teams, index=default_home_idx, format_func=team_with_flag)
away_candidates = [t for t in teams if t != home_team]
away_team = st.selectbox("Away Team", away_candidates, index=0, format_func=team_with_flag)

photo_a, photo_b = st.columns(2)
with photo_a:
    photo = team_photo_path(home_team)
    if photo is not None:
        st.image(str(photo), caption=team_with_flag(home_team), width=260)
with photo_b:
    photo = team_photo_path(away_team)
    if photo is not None:
        st.image(str(photo), caption=team_with_flag(away_team), width=260)

if st.button("Predict Match", type="primary"):
    out = predict_match(home_team=home_team, away_team=away_team)

    home_prob = out["home_win_probability"] * 100.0
    draw_prob = out["draw_probability"] * 100.0
    away_prob = out["away_win_probability"] * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{team_with_flag(home_team)} Win", f"{home_prob:.2f}%")
    c2.metric("Draw", f"{draw_prob:.2f}%")
    c3.metric(f"{team_with_flag(away_team)} Win", f"{away_prob:.2f}%")

    prob_df = pd.DataFrame(
        {
            "Outcome": [f"{team_with_flag(home_team)} Win", "Draw", f"{team_with_flag(away_team)} Win"],
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
    predicted_winner = (
        home_team
        if home_prob >= max(draw_prob, away_prob)
        else away_team
        if away_prob >= max(home_prob, draw_prob)
        else None
    )

    st.subheader("Score and Confidence")
    st.write(
        "Predicted scoreline (expected goals): "
        f"**{team_with_flag(home_team)} {predicted_home_goals:.2f} - {predicted_away_goals:.2f} {team_with_flag(away_team)}**"
    )
    st.write(f"Model confidence (max class probability): **{confidence:.2f}%**")

    if predicted_winner is not None:
        winner_photo = team_photo_path(predicted_winner)
        st.markdown(f"**Projected Match Winner:** {team_with_flag(predicted_winner)}")
        if winner_photo is not None:
            st.image(str(winner_photo), caption=f"Winner Photo: {team_with_flag(predicted_winner)}", width=280)

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
