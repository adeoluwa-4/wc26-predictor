"""Knockout bracket page for World Cup 2026 Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app.dashboard import render_sidebar, run_cached_simulation
from src.app.team_flags import team_with_flag
from src.app.theme import apply_wc26_theme


def _format_knockout_match(row: pd.Series) -> str:
    home = team_with_flag(str(row["home_team"]))
    away = team_with_flag(str(row["away_team"]))
    home_goals = int(row["home_goals"])
    away_goals = int(row["away_goals"])
    winner = team_with_flag(str(row["winner"])) if pd.notna(row.get("winner")) else "TBD"
    pk = " (pens)" if bool(row.get("decided_by_penalties", False)) else ""
    return (
        f"**M{int(row['match_number'])}**  \n"
        f"{home} {home_goals} - {away_goals} {away}{pk}  \n"
        f"Winner: {winner}"
    )


st.set_page_config(page_title="Bracket | World Cup 2026 Predictor", layout="wide")
apply_wc26_theme()

state = render_sidebar(default_team="United States")
outputs = run_cached_simulation(simulations=state.simulations, random_seed=state.random_seed)
knockout = outputs["knockout_matches"]

st.title("Knockout Bracket")
st.caption("This page draws the sample knockout bracket from one simulated tournament run.")

if knockout.empty:
    st.info("No knockout bracket data is available yet. Run a simulation from the sidebar controls.")
    st.stop()

knockout = knockout.copy()
knockout["match_number"] = pd.to_numeric(knockout["match_number"], errors="coerce")
knockout = knockout.dropna(subset=["match_number"]).sort_values("match_number").reset_index(drop=True)

stage_order = [
    ("round_of_32", "Round of 32"),
    ("round_of_16", "Round of 16"),
    ("quarterfinal", "Quarterfinals"),
    ("semifinal", "Semifinals"),
    ("final", "Final"),
]

cols = st.columns(len(stage_order))
for idx, (stage_key, stage_label) in enumerate(stage_order):
    with cols[idx]:
        st.markdown(f"### {stage_label}")
        stage_df = knockout[knockout["stage"] == stage_key]
        if stage_df.empty:
            st.caption("No matches")
            continue
        for _, row in stage_df.iterrows():
            st.markdown(_format_knockout_match(row))

third_place_df = knockout[knockout["stage"] == "third_place"]
if not third_place_df.empty:
    st.markdown("### Third-Place Match")
    for _, row in third_place_df.iterrows():
        st.markdown(_format_knockout_match(row))

with st.expander("Bracket Data (Table)", expanded=False):
    shown = knockout.copy()
    for col in ["home_team", "away_team", "winner"]:
        shown[col] = shown[col].astype(str).map(team_with_flag)
    st.dataframe(
        shown[
            [
                "match_number",
                "stage",
                "home_team",
                "home_goals",
                "away_goals",
                "away_team",
                "winner",
                "decided_by_penalties",
            ]
        ].rename(
            columns={
                "match_number": "Match",
                "stage": "Round",
                "home_team": "Home",
                "home_goals": "HG",
                "away_goals": "AG",
                "away_team": "Away",
                "winner": "Winner",
                "decided_by_penalties": "Pens",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
