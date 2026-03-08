"""Configuration for World Cup simulation."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_GROUP_NAMES = tuple("ABCDEFGHIJKL")


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime settings for tournament simulation."""

    num_groups: int = 12
    group_size: int = 4
    best_third_place_to_advance: int = 8
    points_for_win: int = 3
    points_for_draw: int = 1
    goal_cap: int = 10
    random_seed: int = 42
    default_simulations: int = 1000
    group_names: tuple[str, ...] = DEFAULT_GROUP_NAMES
    min_active_date: str | None = "2018-01-01"
    teams_config_path: str = "data/config/wc26_teams.csv"

    def validate(self) -> None:
        if self.num_groups <= 0:
            raise ValueError("num_groups must be > 0")
        if self.group_size <= 1:
            raise ValueError("group_size must be > 1")
        if self.best_third_place_to_advance < 0:
            raise ValueError("best_third_place_to_advance must be >= 0")

        qualifiers = self.num_groups * 2 + self.best_third_place_to_advance
        if qualifiers <= 1 or qualifiers & (qualifiers - 1) != 0:
            raise ValueError(
                "Qualifiers must be a power of 2. "
                f"Got {qualifiers} from num_groups={self.num_groups} and "
                f"best_third_place_to_advance={self.best_third_place_to_advance}."
            )

        if len(self.group_names) < self.num_groups:
            raise ValueError(
                f"Need at least {self.num_groups} group names, got {len(self.group_names)}"
            )
