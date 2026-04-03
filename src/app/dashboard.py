"""Shared UI helpers for Streamlit pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.team_flags import team_with_flag
from src.models.predict_interface import WC26Predictor
from src.simulation.config import SimulationConfig
from src.simulation.monte_carlo import run_world_cup_simulation
from src.simulation.team_config import load_team_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DashboardState:
    simulations: int
    random_seed: int
    selected_team: str


@st.cache_resource

def get_predictor() -> WC26Predictor:
    """Load trained predictor artifacts once per app process."""
    return WC26Predictor()


@st.cache_data

def get_team_options() -> list[str]:
    """Return confirmed WC teams available in trained profiles.

    Falls back to all model profile teams if fixed config is unavailable.
    """
    predictor = get_predictor()
    profile_teams = set(predictor.team_profiles.reset_index()["team"].dropna().astype(str).tolist())

    cfg = SimulationConfig()
    config_path = Path(cfg.teams_config_path)
    if config_path.exists():
        config_df = load_team_config(config_path, config=cfg)
        confirmed = sorted(
            set(config_df.loc[config_df["status"] == "confirmed", "team"].dropna().astype(str).tolist())
        )
        available = sorted(team for team in confirmed if team in profile_teams)
        missing = sorted(set(confirmed) - set(available))
        if missing:
            LOGGER.warning(
                "Excluding %d confirmed config teams not found in model profiles: %s",
                len(missing),
                missing,
            )
        if available:
            return available

    LOGGER.warning("Falling back to full model-profile team list for UI dropdowns")
    return sorted(profile_teams)


@st.cache_data(show_spinner=True)
def run_cached_simulation(simulations: int, random_seed: int) -> dict[str, object]:
    """Run and cache expensive Monte Carlo simulation."""
    cfg = SimulationConfig(random_seed=random_seed)
    result = run_world_cup_simulation(n_simulations=simulations, config=cfg)

    advancement = result.advancement_probabilities.copy()
    champion = result.champion_probabilities.copy()
    group_winner = result.group_winner_probabilities.copy()
    team_config_rows = pd.DataFrame(result.raw.get("team_config_rows", []))
    projected_rows = pd.DataFrame(
        [row for row in result.raw.get("team_config_rows", []) if row.get("status") == "projected_placeholder"]
    )
    round_of_32_pairings = pd.DataFrame(result.raw.get("sample_run_debug", {}).get("round_of_32_pairings", []))
    knockout_matches = pd.DataFrame(result.raw.get("sample_run_debug", {}).get("knockout_matches", []))
    selected_third_place = pd.DataFrame(result.raw.get("sample_run_debug", {}).get("selected_third_place", []))
    group_finishers = result.raw.get("sample_run_debug", {}).get("group_finishers", {})

    return {
        "advancement": advancement,
        "champion": champion,
        "group_winner": group_winner,
        "team_config": team_config_rows,
        "projected_placeholders": projected_rows,
        "round_of_32_pairings": round_of_32_pairings,
        "knockout_matches": knockout_matches,
        "selected_third_place": selected_third_place,
        "group_finishers": group_finishers,
    }


def render_sidebar(default_team: str | None = None) -> DashboardState:
    """Render shared sidebar controls and return current state."""
    st.sidebar.header("Simulation Controls")
    predictor = get_predictor()
    if getattr(predictor, "_use_fallback_model", False):
        st.sidebar.warning("Using fallback predictor (deployed model binary mismatch).")

    simulations = st.sidebar.slider(
        "Number of simulations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Higher values are slower but more stable.",
    )

    random_seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
        help="Use same seed to reproduce results.",
    )

    teams = get_team_options()
    if not teams:
        raise RuntimeError("No teams found in model profiles. Train models first.")

    if default_team in teams:
        default_idx = teams.index(default_team)
    else:
        default_idx = 0

    selected_team = st.sidebar.selectbox(
        "Selected team",
        teams,
        index=default_idx,
        format_func=team_with_flag,
        help="Used by Team Odds page.",
    )

    st.sidebar.caption("Simulation results are cached by controls above.")

    return DashboardState(simulations=simulations, random_seed=int(random_seed), selected_team=selected_team)
