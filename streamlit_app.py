"""World Cup 2026 Predictor Streamlit entrypoint (Overview page)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from pathlib import Path

from src.app.dashboard import render_sidebar, run_cached_simulation
from src.app.team_images import team_photo_path
from src.app.team_flags import team_with_flag
from src.app.theme import apply_wc26_theme
from src.models.predict_interface import WC26Predictor


st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
apply_wc26_theme()

st.title("World Cup 2026 Predictor")
logo_path = Path("/Users/adeoluwa/Downloads/FIFA-World-Cup-26-Official-Brand-unveiled-in-Los-Angeles.avif")
if logo_path.exists():
    st.image(str(logo_path), caption="World Cup 2026 Official", use_container_width=True)
st.caption("World Cup 2026 prediction dashboard powered by your trained model.")

state = render_sidebar(default_team="United States")
outputs = run_cached_simulation(simulations=state.simulations, random_seed=state.random_seed)

advancement = outputs["advancement"]
champion = outputs["champion"]
round_of_32_pairings = outputs["round_of_32_pairings"]
selected_third_place = outputs["selected_third_place"]
group_finishers = outputs["group_finishers"]

st.caption(
    "Knockout bracket follows the official 2026 FIFA World Cup structure, including best third-place team routing."
)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Simulations", f"{state.simulations:,}")
col_b.metric("Top Favorite", team_with_flag(str(champion.iloc[0]["team"])))
col_c.metric("Top Title Odds", f"{100 * champion.iloc[0]['champion_probability']:.1f}%")

top_team = str(champion.iloc[0]["team"])
top_photo = team_photo_path(top_team)
spot_a, spot_b = st.columns([1, 2])
with spot_a:
    if top_photo is not None:
        st.image(str(top_photo), caption=f"{team_with_flag(top_team)}", width=260)
with spot_b:
    st.subheader("Predicted Champion Spotlight")
    st.write(
        f"Current projected winner: **{team_with_flag(top_team)}** "
        f"({100 * float(champion.iloc[0]['champion_probability']):.2f}% title odds)"
    )

st.subheader("Title Odds")
champion_top10 = champion.head(10).copy()
champion_top10["title_odds_pct"] = champion_top10["champion_probability"] * 100.0
champion_top10["team_display"] = champion_top10["team"].map(team_with_flag)

fig = px.bar(
    champion_top10,
    x="title_odds_pct",
    y="team_display",
    orientation="h",
    title="Top 10 Championship Favorites",
    labels={"title_odds_pct": "Title Odds (%)", "team_display": "Team"},
)
fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=460)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Dark Horses")
predictor = WC26Predictor()
team_elo = predictor.team_profiles.reset_index()[["team", "elo"]].copy()
team_elo["elo_rank"] = team_elo["elo"].rank(ascending=False, method="first")

dark_horses = champion.merge(team_elo, on="team", how="left")
dark_horses = dark_horses[dark_horses["elo_rank"] > 16].copy()
dark_horses = dark_horses.sort_values("champion_probability", ascending=False).head(8)
dark_horses["title_odds_pct"] = dark_horses["champion_probability"] * 100.0

st.dataframe(
    dark_horses.assign(team=dark_horses["team"].map(team_with_flag))[["team", "title_odds_pct", "elo", "elo_rank"]].rename(
        columns={
            "team": "Team",
            "title_odds_pct": "Title Odds (%)",
            "elo": "Elo",
            "elo_rank": "Elo Rank",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Method")
st.markdown(
    """
- Match probabilities come from your trained baseline models.
- Each tournament is simulated match by match with stochastic sampling.
- Group ranking uses points, goal difference, goals for, wins, then random tiebreak.
- Knockout draws are resolved with probability weighted penalty resolution.
"""
)

st.subheader("Quick Sanity")
powerhouse = ["Argentina", "Brazil", "France", "England", "Spain", "Germany"]
sanity = champion[champion["team"].isin(powerhouse)].copy()
sanity["champion_probability"] = sanity["champion_probability"] * 100.0
if sanity.empty:
    st.info("No predefined powerhouse teams present in this simulation pool.")
else:
    st.dataframe(
        sanity.assign(team=sanity["team"].map(team_with_flag)).rename(
            columns={"team": "Team", "champion_probability": "Title Odds (%)"}
        ),
        use_container_width=True,
        hide_index=True,
    )

if "qualified_from_group" in advancement.columns:
    st.caption("Qualification rates and progression rates are available on Team Odds and Group Winners pages.")

with st.expander("Debug: Group Finishers and Knockout Routing", expanded=False):
    if group_finishers:
        finisher_rows = []
        for group_name, row in sorted(group_finishers.items()):
            finisher_rows.append(
                {
                    "Group": group_name,
                    "Winner": team_with_flag(str(row.get("winner"))),
                    "Runner-up": team_with_flag(str(row.get("runner_up"))),
                    "Third": team_with_flag(str(row.get("third"))),
                    "Fourth": team_with_flag(str(row.get("fourth"))),
                }
            )
        st.dataframe(finisher_rows, use_container_width=True, hide_index=True)

    if not selected_third_place.empty:
        st.markdown("**Selected Best Third-Place Teams (sample run)**")
        shown_third = selected_third_place.copy()
        if "team" in shown_third.columns:
            shown_third["team"] = shown_third["team"].map(team_with_flag)
        st.dataframe(shown_third, use_container_width=True, hide_index=True)

    if not round_of_32_pairings.empty:
        st.markdown("**Resolved Round of 32 Pairings (sample run)**")
        shown_r32 = round_of_32_pairings.copy()
        for col in ("home_team", "away_team"):
            if col in shown_r32.columns:
                shown_r32[col] = shown_r32[col].map(team_with_flag)
        st.dataframe(shown_r32, use_container_width=True, hide_index=True)
