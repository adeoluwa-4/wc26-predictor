"""Group winner and qualification probabilities page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.app.dashboard import render_sidebar, run_cached_simulation


st.set_page_config(page_title="Group Winners | WC26 Predictor", layout="wide")

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

fig = px.bar(
    group_df,
    x="team",
    y=["group_winner_probability_pct", "qualification_probability", "third_place_advancement_probability"],
    barmode="group",
    title=f"Group {selected_group} Probabilities",
    labels={"value": "Probability (%)", "team": "Team", "variable": "Metric"},
)
fig.update_layout(height=440)
st.plotly_chart(fig, use_container_width=True)

st.subheader("All Groups Table")
display = merged[[
    "group",
    "team",
    "group_winner_probability_pct",
    "qualification_probability",
    "third_place_advancement_probability",
]].copy()

display = display.rename(
    columns={
        "group": "Group",
        "team": "Team",
        "group_winner_probability_pct": "Group Winner (%)",
        "qualification_probability": "Qualification (%)",
        "third_place_advancement_probability": "Advanced as 3rd (%)",
    }
).sort_values(["Group", "Group Winner (%)"], ascending=[True, False])

st.dataframe(display, use_container_width=True, hide_index=True)
