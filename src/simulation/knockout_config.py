"""Official 2026 knockout slot definitions and deterministic third-place routing."""

from __future__ import annotations

from itertools import combinations


GROUPS = tuple("ABCDEFGHIJKL")

ROUND_OF_32_SLOT_TOKENS: dict[int, tuple[str, str]] = {
    73: ("A2", "B2"),
    74: ("E1", "TP74"),
    75: ("F1", "C2"),
    76: ("C1", "F2"),
    77: ("I1", "TP77"),
    78: ("E2", "I2"),
    79: ("A1", "TP79"),
    80: ("L1", "TP80"),
    81: ("D1", "TP81"),
    82: ("G1", "TP82"),
    83: ("K2", "L2"),
    84: ("H1", "J2"),
    85: ("B1", "TP85"),
    86: ("J1", "H2"),
    87: ("K1", "TP87"),
    88: ("D2", "G2"),
}

THIRD_PLACE_SLOT_ALLOWED_GROUPS: dict[int, frozenset[str]] = {
    74: frozenset(("A", "B", "C", "D", "F")),
    77: frozenset(("C", "D", "F", "G", "H")),
    79: frozenset(("C", "E", "F", "H", "I")),
    80: frozenset(("E", "H", "I", "J", "K")),
    81: frozenset(("B", "E", "F", "I", "J")),
    82: frozenset(("A", "E", "H", "I", "J")),
    85: frozenset(("E", "F", "G", "I", "J")),
    87: frozenset(("D", "E", "I", "J", "L")),
}

THIRD_PLACE_SLOT_ORDER = (74, 77, 79, 80, 81, 82, 85, 87)

ROUND_OF_16_PATHS: dict[int, tuple[int, int]] = {
    89: (73, 75),
    90: (74, 77),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QUARTERFINAL_PATHS: dict[int, tuple[int, int]] = {
    97: (89, 90),
    98: (91, 92),
    99: (93, 94),
    100: (95, 96),
}

SEMIFINAL_PATHS: dict[int, tuple[int, int]] = {
    101: (97, 98),
    102: (99, 100),
}

FINAL_PATH = (101, 102)
THIRD_PLACE_PATH = (101, 102)


def _build_one_combination_mapping(selected_groups: tuple[str, ...]) -> dict[int, str]:
    remaining = set(selected_groups)
    assignment: dict[int, str] = {}

    def feasible(left_slots: tuple[int, ...], left_groups: set[str]) -> bool:
        # Hall-like quick check on all slot subsets.
        for slot in left_slots:
            if not (THIRD_PLACE_SLOT_ALLOWED_GROUPS[slot] & left_groups):
                return False
        return True

    def backtrack(slot_idx: int) -> bool:
        if slot_idx == len(THIRD_PLACE_SLOT_ORDER):
            return True

        slot = THIRD_PLACE_SLOT_ORDER[slot_idx]
        options = sorted(remaining & set(THIRD_PLACE_SLOT_ALLOWED_GROUPS[slot]))
        for group in options:
            assignment[slot] = group
            remaining.remove(group)
            if feasible(THIRD_PLACE_SLOT_ORDER[slot_idx + 1 :], remaining) and backtrack(slot_idx + 1):
                return True
            remaining.add(group)
            del assignment[slot]
        return False

    if not backtrack(0):
        raise ValueError(f"No valid third-place routing for groups: {selected_groups}")

    return assignment


def build_third_place_combination_lookup() -> dict[tuple[str, ...], dict[int, str]]:
    """Build deterministic slot assignments for every 8-group third-place combination."""
    lookup: dict[tuple[str, ...], dict[int, str]] = {}
    for combo in combinations(GROUPS, 8):
        lookup[combo] = _build_one_combination_mapping(combo)
    return lookup


THIRD_PLACE_COMBINATION_LOOKUP = build_third_place_combination_lookup()
