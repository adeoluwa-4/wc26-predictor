"""Shared UI helpers for Streamlit pages."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.models.predict_interface import WC26Predictor
from src.simulation.config import SimulationConfig
from src.simulation.monte_carlo import run_world_cup_simulation


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
    """Return sorted teams available in trained profiles."""
    predictor = get_predictor()
    teams = predictor.team_profiles.reset_index()["team"].dropna().astype(str).tolist()
    return sorted(set(teams))


@st.cache_data(show_spinner=True)

def run_cached_simulation(simulations: int, random_seed: int) -> dict[str, pd.DataFrame]:
    """Run and cache expensive Monte Carlo simulation."""
    cfg = SimulationConfig(random_seed=random_seed)
    result = run_world_cup_simulation(n_simulations=simulations, config=cfg)

    advancement = result.advancement_probabilities.copy()
    champion = result.champion_probabilities.copy()
    group_winner = result.group_winner_probabilities.copy()

    return {
        "advancement": advancement,
        "champion": champion,
        "group_winner": group_winner,
    }


def render_sidebar(default_team: str | None = None) -> DashboardState:
    """Render shared sidebar controls and return current state."""
    st.sidebar.header("Simulation Controls")

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
        help="Used by Team Odds page.",
    )

    st.sidebar.caption("Simulation results are cached by controls above.")

    return DashboardState(simulations=simulations, random_seed=int(random_seed), selected_team=selected_team)
