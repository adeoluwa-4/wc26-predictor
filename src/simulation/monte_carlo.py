"""Monte Carlo simulation runner for WC26 predictor."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from src.models.predict_interface import WC26Predictor
from src.simulation.audit import write_simulation_input_audit
from src.simulation.config import SimulationConfig
from src.simulation.reporting import (
    build_advancement_probabilities,
    build_champion_probabilities,
    build_group_winner_probabilities,
)
from src.simulation.schemas import MonteCarloResult
from src.simulation.team_config import groups_from_team_config, load_team_config
from src.simulation.tournament import build_default_groups, run_single_tournament


def _serialize_knockout_debug(run_debug: object) -> list[dict[str, object]]:
    """Serialize knockout matches with match numbers for UI/debug use."""
    results_by_match = getattr(run_debug, "results_by_match_number", None) or {}
    serialized: list[dict[str, object]] = []
    for match_no, result in sorted(results_by_match.items(), key=lambda item: int(item[0])):
        row = asdict(result)
        row["match_number"] = int(match_no)
        serialized.append(row)
    return serialized


def run_world_cup_simulation(
    n_simulations: int | None = None,
    groups: dict[str, list[str]] | None = None,
    config: SimulationConfig | None = None,
    predict_match_fn: Callable[[str, str], dict[str, float]] | None = None,
) -> MonteCarloResult:
    """Run many full-tournament simulations and aggregate probabilities."""
    cfg = config or SimulationConfig()
    cfg.validate()

    sims = n_simulations or cfg.default_simulations
    if sims <= 0:
        raise ValueError("n_simulations must be > 0")

    predictor: WC26Predictor | None = None
    if predict_match_fn is None:
        predictor = WC26Predictor()
    predict_fn = predict_match_fn or predictor.predict_match

    team_config_df = None
    if groups is not None:
        tournament_groups = groups
    else:
        team_config_path = Path(cfg.teams_config_path)
        if team_config_path.exists():
            team_config_df = load_team_config(team_config_path, config=cfg)
            tournament_groups = groups_from_team_config(team_config_df, config=cfg)
        elif cfg.allow_auto_groups_debug:
            if predictor is None:
                predictor = WC26Predictor()
            tournament_groups = build_default_groups(predictor, config=cfg)
        else:
            raise FileNotFoundError(
                "Fixed group config file is required for simulation. "
                f"Missing: {team_config_path}. "
                "Set allow_auto_groups_debug=True only for debug fallback."
            )

    if team_config_df is not None:
        simulation_input_audit = write_simulation_input_audit(
            output_path=cfg.simulation_input_audit_path,
            groups=tournament_groups,
            team_config=team_config_df,
        )
    else:
        simulation_input_audit = {}

    rng = np.random.default_rng(cfg.random_seed)

    stage_fields = [
        "qualified_from_group",
        "advanced_as_third_place",
        "reached_round_of_32",
        "reached_round_of_16",
        "reached_quarterfinal",
        "reached_semifinal",
        "reached_final",
        "won_tournament",
    ]

    teams = sorted({team for team_list in tournament_groups.values() for team in team_list})
    progression_counts: dict[str, dict[str, int]] = {
        team: {field: 0 for field in stage_fields} for team in teams
    }

    group_winner_counts: dict[tuple[str, str], int] = defaultdict(int)
    sample_debug: dict[str, object] = {}

    for sim_idx in range(sims):
        seed = int(rng.integers(0, 2**32 - 1))
        run = run_single_tournament(
            groups=tournament_groups,
            predict_match_fn=predict_fn,
            seed=seed,
            config=cfg,
            include_group_details=(sim_idx == 0),
        )

        for team, flags in run.progression.items():
            for field, flag in flags.items():
                if flag:
                    progression_counts[team][field] += 1

        for group_name, winner in run.group_winners.items():
            group_winner_counts[(group_name, winner)] += 1

        if sim_idx == 0:
            sample_debug = {
                "group_finishers": run.group_finishers,
                "selected_third_place": (
                    run.selected_third_place.to_dict(orient="records")
                    if run.selected_third_place is not None
                    else []
                ),
                "third_place_slot_groups": run.third_place_slot_groups or {},
                "round_of_32_pairings": run.round_of_32_pairings or [],
                "knockout_matches": _serialize_knockout_debug(run.knockout),
            }

    advancement_df = build_advancement_probabilities(progression_counts=progression_counts, simulations=sims)
    champion_df = build_champion_probabilities(advancement_probabilities=advancement_df)
    group_winner_df = build_group_winner_probabilities(group_winner_counts=group_winner_counts, simulations=sims)

    return MonteCarloResult(
        advancement_probabilities=advancement_df,
        champion_probabilities=champion_df,
        group_winner_probabilities=group_winner_df,
        raw={
            "simulations": sims,
            "groups": tournament_groups,
            "progression_counts": progression_counts,
            "group_winner_counts": {f"{g}:{t}": c for (g, t), c in group_winner_counts.items()},
            "team_config_rows": (
                team_config_df.to_dict(orient="records") if team_config_df is not None else []
            ),
            "simulation_input_audit": simulation_input_audit,
            "sample_run_debug": sample_debug,
        },
    )
