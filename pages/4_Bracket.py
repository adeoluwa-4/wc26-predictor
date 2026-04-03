"""Knockout bracket page for World Cup 2026 Streamlit app."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.app.dashboard import render_sidebar, run_cached_simulation
from src.app.team_flags import team_flag
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
    return value if len(value) <= 14 else f"{value[:13]}…"


def _team_label(team: str) -> str:
    return f"{team_flag(team)} {_short_team(team)}"


def _build_bracket_svg(knockout: pd.DataFrame) -> str:
    rows = {int(r["match_number"]): r for _, r in knockout.iterrows()}

    left_r32 = [73, 74, 75, 76, 77, 78, 79, 80]
    right_r32 = [81, 82, 83, 84, 85, 86, 87, 88]
    left_r16 = [89, 90, 91, 92]
    right_r16 = [93, 94, 95, 96]
    left_qf = [97, 98]
    right_qf = [99, 100]
    left_sf = [101]
    right_sf = [102]
    final = [103]

    card_w = 188
    card_h = 42
    top = 100
    row_gap = 80

    left_x = {"r32": 24, "r16": 246, "qf": 468, "sf": 690}
    right_x = {"r32": 1708, "r16": 1486, "qf": 1264, "sf": 1042}
    final_x = 866

    x_by_match: dict[int, float] = {}
    y_by_match: dict[int, float] = {}

    for m in left_r32:
        x_by_match[m] = left_x["r32"]
    for m in right_r32:
        x_by_match[m] = right_x["r32"]
    for m in left_r16:
        x_by_match[m] = left_x["r16"]
    for m in right_r16:
        x_by_match[m] = right_x["r16"]
    for m in left_qf:
        x_by_match[m] = left_x["qf"]
    for m in right_qf:
        x_by_match[m] = right_x["qf"]
    for m in left_sf:
        x_by_match[m] = left_x["sf"]
    for m in right_sf:
        x_by_match[m] = right_x["sf"]
    x_by_match[103] = final_x

    for idx, m in enumerate(left_r32):
        y_by_match[m] = top + idx * row_gap
    for idx, m in enumerate(right_r32):
        y_by_match[m] = top + idx * row_gap

    for match_id, (left, right) in ROUND_OF_16_PATHS.items():
        y_by_match[match_id] = (y_by_match[left] + y_by_match[right]) / 2.0
    for match_id, (left, right) in QUARTERFINAL_PATHS.items():
        y_by_match[match_id] = (y_by_match[left] + y_by_match[right]) / 2.0
    for match_id, (left, right) in SEMIFINAL_PATHS.items():
        y_by_match[match_id] = (y_by_match[left] + y_by_match[right]) / 2.0
    y_by_match[103] = (y_by_match[FINAL_PATH[0]] + y_by_match[FINAL_PATH[1]]) / 2.0

    width = int(right_x["r32"] + card_w + 28)
    height = int(max(y_by_match.values()) + 125)

    out: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#f3f4f6"/>',
        '<text x="970" y="36" fill="#4b5563" font-size="28" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-weight="700" letter-spacing="3">WORLD CUP 2026</text>',
    ]

    stage_titles = [
        (left_x["r32"] + card_w / 2.0, "Round of 32"),
        (left_x["r16"] + card_w / 2.0, "Round of 16"),
        (left_x["qf"] + card_w / 2.0, "Quarterfinals"),
        (left_x["sf"] + card_w / 2.0, "Semifinal"),
        (final_x + card_w / 2.0, "Final"),
        (right_x["sf"] + card_w / 2.0, "Semifinal"),
        (right_x["qf"] + card_w / 2.0, "Quarterfinals"),
        (right_x["r16"] + card_w / 2.0, "Round of 16"),
        (right_x["r32"] + card_w / 2.0, "Round of 32"),
    ]
    for x, title in stage_titles:
        out.append(
            f'<text x="{x:.1f}" y="66" fill="#6b7280" font-size="12" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-weight="600">{escape(title)}</text>'
        )

    def connect_left(src_id: int, target_id: int) -> None:
        x1 = x_by_match[src_id] + card_w
        y1 = y_by_match[src_id]
        x2 = x_by_match[target_id]
        y2 = y_by_match[target_id]
        mid = (x1 + x2) / 2.0
        out.append(
            f'<path d="M {x1:.1f} {y1:.1f} H {mid:.1f} V {y2:.1f} H {x2:.1f}" '
            'stroke="#9ca3af" stroke-width="1.6" fill="none"/>'
        )

    def connect_right(src_id: int, target_id: int) -> None:
        x1 = x_by_match[src_id]
        y1 = y_by_match[src_id]
        x2 = x_by_match[target_id] + card_w
        y2 = y_by_match[target_id]
        mid = (x1 + x2) / 2.0
        out.append(
            f'<path d="M {x1:.1f} {y1:.1f} H {mid:.1f} V {y2:.1f} H {x2:.1f}" '
            'stroke="#9ca3af" stroke-width="1.6" fill="none"/>'
        )

    left_targets = set(left_r16 + left_qf + left_sf + final)
    right_targets = set(right_r16 + right_qf + right_sf + final)

    for target, (a, b) in ROUND_OF_16_PATHS.items():
        if target in left_targets:
            connect_left(a, target)
            connect_left(b, target)
        else:
            connect_right(a, target)
            connect_right(b, target)

    for target, (a, b) in QUARTERFINAL_PATHS.items():
        if target in left_targets:
            connect_left(a, target)
            connect_left(b, target)
        else:
            connect_right(a, target)
            connect_right(b, target)

    for target, (a, b) in SEMIFINAL_PATHS.items():
        if target in left_targets:
            connect_left(a, target)
            connect_left(b, target)
        else:
            connect_right(a, target)
            connect_right(b, target)

    connect_left(FINAL_PATH[0], 103)
    connect_right(FINAL_PATH[1], 103)

    ordered = left_r32 + right_r32 + left_r16 + right_r16 + left_qf + right_qf + left_sf + right_sf + final
    for match_id in ordered:
        row = rows.get(match_id)
        if row is None:
            continue
        x = x_by_match[match_id]
        y = y_by_match[match_id]
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home = _team_label(home_team)
        away = _team_label(away_team)
        winner = str(row.get("winner") or "")
        home_w = "700" if winner and winner == home_team else "400"
        away_w = "700" if winner and winner == away_team else "400"
        card_y = y - (card_h / 2.0)
        out.append(
            f'<rect x="{x:.1f}" y="{card_y:.1f}" width="{card_w}" height="{card_h}" rx="7" ry="7" '
            'fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>'
        )
        out.append(
            f'<text x="{x + 8:.1f}" y="{card_y + 15:.1f}" fill="#111827" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{home_w}">{escape(home)}</text>'
        )
        out.append(
            f'<text x="{x + 8:.1f}" y="{card_y + 31:.1f}" fill="#111827" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{away_w}">{escape(away)}</text>'
        )

    if 104 in rows:
        row = rows[104]
        x = final_x
        y = max(y_by_match.values()) + 64
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home = _team_label(home_team)
        away = _team_label(away_team)
        winner = str(row.get("winner") or "")
        home_w = "700" if winner and winner == home_team else "400"
        away_w = "700" if winner and winner == away_team else "400"
        card_y = y - (card_h / 2.0)
        out.append(
            f'<text x="{x + card_w / 2.0:.1f}" y="{card_y - 10:.1f}" fill="#6b7280" font-size="12" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-weight="600">Third Place</text>'
        )
        out.append(
            f'<rect x="{x:.1f}" y="{card_y:.1f}" width="{card_w}" height="{card_h}" rx="7" ry="7" '
            'fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>'
        )
        out.append(
            f'<text x="{x + 8:.1f}" y="{card_y + 15:.1f}" fill="#111827" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{home_w}">{escape(home)}</text>'
        )
        out.append(
            f'<text x="{x + 8:.1f}" y="{card_y + 31:.1f}" fill="#111827" font-size="11" '
            f'font-family="Arial, sans-serif" font-weight="{away_w}">{escape(away)}</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


st.set_page_config(page_title="Bracket | World Cup 2026 Predictor", layout="wide")
apply_wc26_theme()

state = render_sidebar(default_team="United States")
outputs = run_cached_simulation(simulations=state.simulations, random_seed=state.random_seed)
knockout = outputs["knockout_matches"]

st.title("World Cup 2026 Bracket")

if knockout.empty:
    st.info("No bracket data available yet.")
    st.stop()

knockout = knockout.copy()
knockout["match_number"] = pd.to_numeric(knockout["match_number"], errors="coerce")
knockout = knockout.dropna(subset=["match_number"]).sort_values("match_number").reset_index(drop=True)

svg = _build_bracket_svg(knockout)
components.html(
    f"""
<div style="overflow:auto; width:100%; border:1px solid #cbd5e1; border-radius:12px; background:#f3f4f6;">
{svg}
</div>
""",
    height=900,
    scrolling=True,
)
