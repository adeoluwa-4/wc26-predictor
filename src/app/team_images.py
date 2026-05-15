"""Team image lookup helpers for Streamlit UI."""

from __future__ import annotations

from pathlib import Path
import re

from src.utils.paths import PROJECT_ROOT


TEAM_PHOTO_DIR = PROJECT_ROOT / "assets" / "team_photos"
PHOTO_EXTENSIONS = (".avif", ".png", ".jpg", ".jpeg", ".webp")

TEAM_FILE_OVERRIDES = {
    "United States": "USA",
    "South Korea": "Korea",
    "Bosnia and Herzegovina": "Bosnia",
    "Ivory Coast": "IvoryCoast",
    "DR Congo": "DRCongo",
    "Czech Republic": "CzechRepublic",
    "Cape Verde": "CapeVerde",
    "Saudi Arabia": "SaudiArabia",
    "New Zealand": "NewZealand",
}


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "", value.strip())
    return clean


def team_photo_path(team: str) -> Path | None:
    """Return best-effort local photo path for a team, if available."""
    if not team:
        return None

    candidates = []
    override = TEAM_FILE_OVERRIDES.get(team)
    if override:
        candidates.append(override)
    candidates.extend([team, _slug(team), team.replace(" ", "_"), team.replace(" ", "-")])

    seen: set[str] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        for ext in PHOTO_EXTENSIONS:
            p = TEAM_PHOTO_DIR / f"{base}{ext}"
            if p.exists():
                return p
    return None
