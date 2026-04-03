"""Group winner and qualification probabilities page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.app.dashboard import render_sidebar, run_cached_simulation
from src.app.team_flags import team_with_flag
from src.app.theme import apply_wc26_theme


st.set_page_config(page_title="Group Winners | World Cup 2026 Predictor", layout="wide")
apply_wc26_theme()

state = render_sidebar(default_team="United States")
outputs = run_cached_simulation(simulations=state.simulations, random_seed=state.random_seed)
advancement = outputs["advancement"]
group_winner = outputs["group_winner"]

st.title("Group Winners")
st.caption("Winner odds, group qualification odds, and third-place advancement rates.")

if group_winner.empty:
    st.warning("No group winner data available. Run simulations first.")
    st.stop()

merged = group_winner.merge(
    advancement[[
        "team",
        *[c for c in ["qualified_from_group", "advanced_as_third_place", "reached_round_of_32"] if c in advancement.columns],
    ]],
    on="team",
    how="left",
)

if "qualified_from_group" in merged.columns:
    merged["qualification_probability"] = merged["qualified_from_group"] * 100.0
elif "reached_round_of_32" in merged.columns:
    merged["qualification_probability"] = merged["reached_round_of_32"] * 100.0
else:
    merged["qualification_probability"] = 0.0

if "advanced_as_third_place" in merged.columns:
    merged["third_place_advancement_probability"] = merged["advanced_as_third_place"] * 100.0
else:
    merged["third_place_advancement_probability"] = 0.0

merged["group_winner_probability_pct"] = merged["group_winner_probability"] * 100.0

groups = sorted(merged["group"].unique().tolist())
selected_group = st.selectbox("Select group", groups)

group_df = merged[merged["group"] == selected_group].sort_values("group_winner_probability_pct", ascending=False)
group_df = group_df.assign(team_display=group_df["team"].map(team_with_flag))

fig = px.bar(
    group_df,
    x="team_display",
    y=["group_winner_probability_pct", "qualification_probability", "third_place_advancement_probability"],
    barmode="group",
    title=f"Group {selected_group} Probabilities",
    labels={"value": "Probability (%)", "team_display": "Team", "variable": "Metric"},
)
fig.update_layout(height=440)
st.plotly_chart(fig, use_container_width=True)

st.subheader("All Groups Table")
display = merged.copy()
display["team"] = display["team"].map(team_with_flag)
display = display.sort_values(["group", "group_winner_probability_pct"], ascending=[True, False])

for i in range(0, len(groups), 3):
    row_groups = groups[i:i + 3]
    cols = st.columns(3)
    for j in range(3):
        with cols[j]:
            if j >= len(row_groups):
                st.empty()
                continue
            group_name = row_groups[j]
            gdf = display[display["group"] == group_name][
                [
                    "team",
                    "group_winner_probability_pct",
                    "qualification_probability",
                    "third_place_advancement_probability",
                ]
            ].copy()
            gdf = gdf.rename(
                columns={
                    "team": "Team",
                    "group_winner_probability_pct": "Winner %",
                    "qualification_probability": "Qualify %",
                    "third_place_advancement_probability": "3rd %",
                }
            )
            st.markdown(f"### Group {group_name}")
            st.dataframe(gdf, use_container_width=True, hide_index=True)
