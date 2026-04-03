"""Team odds page for World Cup 2026 Streamlit app."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.dashboard import get_team_options, render_sidebar, run_cached_simulation
from src.app.team_images import team_photo_path
from src.app.team_flags import team_flag, team_with_flag
from src.app.theme import apply_wc26_theme


st.set_page_config(page_title="Team Odds | World Cup 2026 Predictor", layout="wide")
apply_wc26_theme()

state = render_sidebar(default_team="United States")
outputs = run_cached_simulation(simulations=state.simulations, random_seed=state.random_seed)
advancement = outputs["advancement"]
champion = outputs["champion"]

st.title("Team Odds")
st.markdown(
    """
<style>
.wc26-photo-placeholder {
    width: 100%;
    aspect-ratio: 4 / 3;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 44px;
    background: linear-gradient(135deg, rgba(10, 90, 160, 0.35), rgba(220, 10, 40, 0.35));
    border: 1px solid rgba(255, 255, 255, 0.2);
}
</style>
""",
    unsafe_allow_html=True,
)

teams = get_team_options()
selected_team = st.selectbox(
    "Choose a team",
    teams,
    index=teams.index(state.selected_team) if state.selected_team in teams else 0,
    format_func=team_with_flag,
)
photo = team_photo_path(selected_team)
if photo is not None:
    st.image(str(photo), caption=team_with_flag(selected_team), width=260)

row_df = advancement[advancement["team"] == selected_team]
if row_df.empty:
    st.warning(f"No simulation row found for {selected_team}.")
    st.stop()

row = row_df.iloc[0]
champ_row = champion[champion["team"] == selected_team]
champ_prob = float(champ_row.iloc[0]["champion_probability"]) if not champ_row.empty else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Title Odds", f"{champ_prob * 100:.2f}%")
c2.metric("Final", f"{float(row.get('reached_final', 0.0)) * 100:.2f}%")
c3.metric("Semifinal", f"{float(row.get('reached_semifinal', 0.0)) * 100:.2f}%")
c4.metric("Quarterfinal", f"{float(row.get('reached_quarterfinal', 0.0)) * 100:.2f}%")

progression_keys = [
    ("qualified_from_group", "Qualified"),
    ("advanced_as_third_place", "Advanced as 3rd"),
    ("reached_round_of_32", "Round of 32"),
    ("reached_round_of_16", "Round of 16"),
    ("reached_quarterfinal", "Quarterfinal"),
    ("reached_semifinal", "Semifinal"),
    ("reached_final", "Final"),
    ("won_tournament", "Champion"),
]

plot_rows = []
for key, label in progression_keys:
    if key in advancement.columns:
        plot_rows.append({"stage": label, "probability": float(row[key]) * 100.0})

plot_df = pd.DataFrame(plot_rows)

fig = px.bar(
    plot_df,
    x="stage",
    y="probability",
    title=f"{team_with_flag(selected_team)} Advancement Probabilities",
    labels={"probability": "Probability (%)", "stage": "Stage"},
)
fig.update_layout(height=430)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Where This Team Ranks")
champion_rank = champion.reset_index(drop=True)
champion_rank["rank"] = champion_rank.index + 1
entry = champion_rank[champion_rank["team"] == selected_team]
if not entry.empty:
    rank = int(entry.iloc[0]["rank"])
    st.write(f"Current title-odds rank: **#{rank}** out of {len(champion_rank)} teams.")

st.dataframe(
    champion_rank.head(20)
    .assign(team=lambda d: d["team"].map(team_with_flag))
    .rename(columns={"team": "Team", "champion_probability": "Champion Probability"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("All Teams Photo Wall")
odds_map = {str(row["team"]): float(row["champion_probability"]) for _, row in champion.iterrows()}
photo_teams = [str(team) for team in champion["team"].tolist()]
gallery_cols = st.columns(6)
for idx, team in enumerate(photo_teams):
    with gallery_cols[idx % 6]:
        image_path = team_photo_path(team)
        if image_path is not None:
            st.image(str(image_path), use_container_width=True)
        else:
            st.markdown(f"<div class='wc26-photo-placeholder'>{team_flag(team)}</div>", unsafe_allow_html=True)
        st.caption(team_with_flag(team))
        st.caption(f"Title odds: {odds_map.get(team, 0.0) * 100:.2f}%")
