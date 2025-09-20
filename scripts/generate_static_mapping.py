"""Generate a static player->team mapping for a legacy season.

Approach:
  - Load players file from a reference (modern) season (default: 2025-2026)
  - Extract (player_id, team_code) and emit as (id, team_id) for target legacy season (default: 2024-2025)

Environment variables:
  REFERENCE_SEASON (default 2025-2026)
  TARGET_SEASON (default 2024-2025)
  OUTPUT_FILENAME (default player_team_mapping.csv)

Assumptions:
  - Player IDs are stable across seasons.
  - team_code in reference season is a valid surrogate for team_id for legacy season match context mapping.

Limitations:
  - Ignores mid-season transfers.
  - Players absent in reference season will not receive mapping.
"""
from __future__ import annotations
import os
import sys
import pandas as pd

REFERENCE_SEASON = os.environ.get("REFERENCE_SEASON", "2025-2026")
TARGET_SEASON = os.environ.get("TARGET_SEASON", "2024-2025")
OUTPUT_FILENAME = os.environ.get("OUTPUT_FILENAME", "player_team_mapping.csv")

REF_PLAYERS_PATH = os.path.join("data", REFERENCE_SEASON, "players.csv")
TARGET_DIR = os.path.join("data", TARGET_SEASON)
OUTPUT_PATH = os.path.join(TARGET_DIR, OUTPUT_FILENAME)


def log(msg: str):
    print(msg, flush=True)

def main():
    log(f"--- Generating static player->team mapping for {TARGET_SEASON} from {REFERENCE_SEASON} ---")
    if not os.path.exists(REF_PLAYERS_PATH):
        log(f"❌ Reference players file not found: {REF_PLAYERS_PATH}")
        sys.exit(1)
    try:
        ref = pd.read_csv(REF_PLAYERS_PATH)
    except Exception as e:
        log(f"❌ Failed reading reference players: {e}")
        sys.exit(1)

    needed_cols = ["player_id", "team_code"]
    missing = [c for c in needed_cols if c not in ref.columns]
    if missing:
        log(f"❌ Missing required columns in reference players: {missing}")
        sys.exit(1)

    mapping = ref[["player_id", "team_code"]].drop_duplicates().rename(columns={"player_id": "id", "team_code": "team_id"})
    os.makedirs(TARGET_DIR, exist_ok=True)
    mapping.to_csv(OUTPUT_PATH, index=False)
    log(f"✅ Wrote {len(mapping)} player mappings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
