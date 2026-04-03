"""Knockout bracket page for World Cup 2026 Streamlit app."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.app.dashboard import render_sidebar, run_cached_simulation
from src.app.team_flags import team_with_flag
from src.app.theme import apply_wc26_theme
from src.simulation.knockout_config import FINAL_PATH, QUARTERFINAL_PATHS, ROUND_OF_16_PATHS, SEMIFINAL_PATHS


def _short_team(team: str) -> str:
    aliases = {
        "Bosnia and Herzegovina": "Bosnia",
        "Czech Republic": "Czechia",
        "South Korea": "Korea",
        "Saudi Arabia": "Saudi",
        "New Zealand": "N. Zealand",
        "Ivory Coast": "Ivory Coast",
        "United States": "USA",
    }
    value = aliases.get(team, team)
    if len(value) <= 14:
        return value
    return f"{value[:13]}…"


def _build_bracket_svg(knockout: pd.DataFrame) -> str:
    rows = {int(r["match_number"]): r for _, r in knockout.iterrows()}
    r32 = list(range(73, 89))
    r16 = sorted(ROUND_OF_16_PATHS.keys())
    qf = sorted(QUARTERFINAL_PATHS.keys())
    sf = sorted(SEMIFINAL_PATHS.keys())
    final = [103]

    stage_ids = [r32, r16, qf, sf, final]
    stage_titles = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]

    card_w = 190
    card_h = 42
    top = 90
    row_gap = 68
    col_gap = 250
    x_start = 24

    x_by_match: dict[int, float] = {}
    y_by_match: dict[int, float] = {}
    stage_x: list[float] = []

    for idx, ids in enumerate(stage_ids):
        x = x_start + idx * col_gap
        stage_x.append(x)
        for match_id in ids:
            x_by_match[match_id] = x

    for idx, match_id in enumerate(r32):
        y_by_match[match_id] = top + idx * row_gap
    for match_id, (left, right) in ROUND_OF_16_PATHS.items():
        y_by_match[match_id] = (y_by_match[left] + y_by_match[right]) / 2.0
    for match_id, (left, right) in QUARTERFINAL_PATHS.items():
        y_by_match[match_id] = (y_by_match[left] + y_by_match[right]) / 2.0
    for match_id, (left, right) in SEMIFINAL_PATHS.items():
        y_by_match[match_id] = (y_by_match[left] + y_by_match[right]) / 2.0
    y_by_match[103] = (y_by_match[FINAL_PATH[0]] + y_by_match[FINAL_PATH[1]]) / 2.0

    width = int(stage_x[-1] + card_w + 40)
    height = int(max(y_by_match.values()) + 130)

    out: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#0a0f1f"/>',
    ]

    for x, title in zip(stage_x, stage_titles, strict=True):
        out.append(
            f'<text x="{x + card_w / 2:.1f}" y="34" fill="#cfd8e3" font-size="17" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-weight="600">{escape(title)}</text>'
        )

    def connect(left_id: int, right_id: int, target_id: int) -> None:
        for src_id in (left_id, right_id):
            x1 = x_by_match[src_id] + card_w
            y1 = y_by_match[src_id]
            x2 = x_by_match[target_id]
            y2 = y_by_match[target_id]
            mid = (x1 + x2) / 2.0
            out.append(
                f'<path d="M {x1:.1f} {y1:.1f} H {mid:.1f} V {y2:.1f} H {x2:.1f}" '
                'stroke="#6f7ea7" stroke-width="1.5" fill="none"/>'
            )

    for target, (left, right) in ROUND_OF_16_PATHS.items():
        connect(left, right, target)
    for target, (left, right) in QUARTERFINAL_PATHS.items():
        connect(left, right, target)
    for target, (left, right) in SEMIFINAL_PATHS.items():
        connect(left, right, target)
    connect(FINAL_PATH[0], FINAL_PATH[1], 103)

    all_main_matches = r32 + r16 + qf + sf + final
    for match_id in all_main_matches:
        row = rows.get(match_id)
        if row is None:
            continue
        x = x_by_match[match_id]
        y = y_by_match[match_id]
        home = _short_team(str(row["home_team"]))
        away = _short_team(str(row["away_team"]))
        home_score = int(row["home_goals"])
        away_score = int(row["away_goals"])
        winner = str(row.get("winner") or "")
        home_bold = "700" if winner and winner == str(row["home_team"]) else "400"
        away_bold = "700" if winner and winner == str(row["away_team"]) else "400"
        card_x = x
        card_y = y - (card_h / 2)
        out.append(
            f'<rect x="{card_x:.1f}" y="{card_y:.1f}" width="{card_w}" height="{card_h}" rx="8" ry="8" '
            'fill="#121a31" stroke="#31466f" stroke-width="1.2"/>'
        )
        out.append(
            f'<text x="{card_x + 8:.1f}" y="{card_y + 15:.1f}" fill="#eaf0f8" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{home_bold}">{escape(home)} {home_score}</text>'
        )
        out.append(
            f'<text x="{card_x + 8:.1f}" y="{card_y + 32:.1f}" fill="#eaf0f8" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{away_bold}">{escape(away)} {away_score}</text>'
        )
        out.append(
            f'<text x="{card_x + card_w - 8:.1f}" y="{card_y + 14:.1f}" fill="#8ea3c4" font-size="9" text-anchor="end" '
            f'font-family="Arial, sans-serif">M{match_id}</text>'
        )

    if 104 in rows:
        row = rows[104]
        x = stage_x[-2]
        y = max(y_by_match.values()) + 70
        home = _short_team(str(row["home_team"]))
        away = _short_team(str(row["away_team"]))
        home_score = int(row["home_goals"])
        away_score = int(row["away_goals"])
        winner = str(row.get("winner") or "")
        home_bold = "700" if winner and winner == str(row["home_team"]) else "400"
        away_bold = "700" if winner and winner == str(row["away_team"]) else "400"
        card_x = x
        card_y = y - (card_h / 2)
        out.append(
            f'<text x="{card_x + card_w / 2:.1f}" y="{card_y - 12:.1f}" fill="#cfd8e3" font-size="14" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-weight="600">Third Place</text>'
        )
        out.append(
            f'<rect x="{card_x:.1f}" y="{card_y:.1f}" width="{card_w}" height="{card_h}" rx="8" ry="8" '
            'fill="#121a31" stroke="#31466f" stroke-width="1.2"/>'
        )
        out.append(
            f'<text x="{card_x + 8:.1f}" y="{card_y + 15:.1f}" fill="#eaf0f8" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{home_bold}">{escape(home)} {home_score}</text>'
        )
        out.append(
            f'<text x="{card_x + 8:.1f}" y="{card_y + 32:.1f}" fill="#eaf0f8" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{away_bold}">{escape(away)} {away_score}</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


st.set_page_config(page_title="Bracket | World Cup 2026 Predictor", layout="wide")
apply_wc26_theme()

state = render_sidebar(default_team="United States")
outputs = run_cached_simulation(simulations=state.simulations, random_seed=state.random_seed)
knockout = outputs["knockout_matches"]

st.title("Knockout Bracket")
st.caption("Line-connected bracket from one sample tournament run.")

if knockout.empty:
    st.info("No knockout bracket data is available yet. Run a simulation from the sidebar controls.")
    st.stop()

knockout = knockout.copy()
knockout["match_number"] = pd.to_numeric(knockout["match_number"], errors="coerce")
knockout = knockout.dropna(subset=["match_number"]).sort_values("match_number").reset_index(drop=True)

svg = _build_bracket_svg(knockout)
components.html(
    f"""
<div style="overflow:auto; width:100%; border:1px solid #2d3c5f; border-radius:12px; background:#0a0f1f;">
{svg}
</div>
""",
    height=900,
    scrolling=True,
)

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
