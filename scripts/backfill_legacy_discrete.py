"""Backfill per-gameweek discrete player stats for a legacy season.

Legacy season (2024-2025) stores cumulative `playerstats/playerstats.csv` and
per-GW match & playermatchstats folders, but lacks the modern `By Gameweek` tree
with `player_gameweek_stats.csv` delta files.

This script reconstructs the modern structure:
  data/<season>/By Gameweek/GW{n}/
    - playerstats.csv (cumulative slice for GW)
    - player_gameweek_stats.csv (discrete deltas vs previous GW)
    - matches.csv (copied from legacy matches/GW{n}/matches.csv)
    - fixtures.csv (duplicate of matches.csv for structural parity)
    - playermatchstats.csv (copied from legacy playermatchstats/GW{n}/playermatchstats.csv if exists)

It mirrors the delta logic in export_data.calculate_discrete_gameweek_stats().

Environment variable LEGACY_SEASON can override the default season.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
from typing import List

LEGACY_SEASON = os.environ.get("LEGACY_SEASON", "2024-2025")
BASE_PATH = os.path.join("data", LEGACY_SEASON)
PLAYERSTATS_PATH = os.path.join(BASE_PATH, "playerstats", "playerstats.csv")
MATCHES_ROOT = os.path.join(BASE_PATH, "matches")
PLAYERMATCH_ROOT = os.path.join(BASE_PATH, "playermatchstats")
OUTPUT_ROOT = os.path.join(BASE_PATH, "By Gameweek")

# Reuse the cumulative columns definition from modern pipeline (mirror / subset)
CUMULATIVE_COLS = [
    'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
    'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed',
    'yellow_cards', 'red_cards', 'saves', 'starts', 'bonus', 'bps',
    'transfers_in', 'transfers_out', 'dreamteam_count', 'expected_goals',
    'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
    'influence', 'creativity', 'threat', 'ict_index'
]

ID_COLS = ['id', 'first_name', 'second_name', 'web_name']  # Will be partially missing in legacy; safe subset
SNAPSHOT_COLS = [
    'status', 'news', 'now_cost', 'selected_by_percent', 'form', 'event_points',
    'cost_change_event', 'transfers_in_event', 'transfers_out_event',
    'value_form', 'value_season', 'ep_next', 'ep_this'
]


def log(msg: str):
    print(msg, flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    log(f"--- Backfilling discrete stats for legacy season {LEGACY_SEASON} ---")

    if not os.path.exists(PLAYERSTATS_PATH):
        log(f"❌ playerstats file not found: {PLAYERSTATS_PATH}")
        sys.exit(1)

    ps = pd.read_csv(PLAYERSTATS_PATH)
    if 'gw' not in ps.columns:
        log("❌ Legacy playerstats missing 'gw' column; cannot segment by gameweek.")
        sys.exit(1)

    # Determine ordered gameweeks
    gameweeks = sorted(ps['gw'].dropna().unique())
    log(f"Found {len(gameweeks)} gameweeks: {gameweeks[:5]}{'...' if len(gameweeks)>5 else ''}")

    # Build mapping of available columns
    available_cumulatives = [c for c in CUMULATIVE_COLS if c in ps.columns]
    available_snapshots = [c for c in SNAPSHOT_COLS if c in ps.columns]
    available_ids = [c for c in ID_COLS if c in ps.columns]

    prev_df = None
    rows_written = 0

    for gw in gameweeks:
        gw_int = int(gw)
        gw_slice = ps[ps['gw'] == gw].copy()
        gw_folder = os.path.join(OUTPUT_ROOT, f"GW{gw_int}")
        ensure_dir(gw_folder)

        # Write cumulative slice as playerstats.csv (aligned with modern structure subset)
        gw_slice.to_csv(os.path.join(gw_folder, 'playerstats.csv'), index=False)

        if prev_df is None:
            # Baseline: raw values for cumulatives + snapshots + ids
            baseline_cols = available_ids + available_snapshots + available_cumulatives
            existing = [c for c in baseline_cols if c in gw_slice.columns]
            discrete = gw_slice[existing].copy()
        else:
            # Merge previous GW cumulative values to subtract
            merge_cols_prev = [c for c in available_ids + available_cumulatives if c in prev_df.columns]
            merge_cols_curr = [c for c in available_ids + available_cumulatives if c in gw_slice.columns]
            prev_sub = prev_df[merge_cols_prev].copy()
            curr_sub = gw_slice[merge_cols_curr].copy()
            merged = pd.merge(curr_sub, prev_sub, on='id', how='left', suffixes=('', '_prev'))
            for col in available_cumulatives:
                if col in merged.columns and f"{col}_prev" in merged.columns:
                    merged[f"{col}_prev"].fillna(0, inplace=True)
                    merged[col] = merged[col] - merged[f"{col}_prev"]
            final_cols = available_ids + available_snapshots + available_cumulatives
            existing = [c for c in final_cols if c in merged.columns]
            discrete = merged[existing].copy()

        # Save discrete deltas
        discrete.to_csv(os.path.join(gw_folder, 'player_gameweek_stats.csv'), index=False)
        rows_written += len(discrete)

        # Copy matches if present
        legacy_match_path = os.path.join(MATCHES_ROOT, f"GW{gw_int}", 'matches.csv')
        if os.path.exists(legacy_match_path):
            try:
                match_df = pd.read_csv(legacy_match_path)
                match_df.to_csv(os.path.join(gw_folder, 'matches.csv'), index=False)
                match_df.to_csv(os.path.join(gw_folder, 'fixtures.csv'), index=False)
            except Exception as e:
                log(f"⚠️  GW{gw_int}: failed to copy matches - {e}")
        else:
            log(f"⚠️  GW{gw_int}: matches.csv not found at legacy path, skipping matches/fixtures copy.")

        # Copy playermatchstats if present
        legacy_pms_path = os.path.join(PLAYERMATCH_ROOT, f"GW{gw_int}", 'playermatchstats.csv')
        if os.path.exists(legacy_pms_path):
            try:
                pms_df = pd.read_csv(legacy_pms_path)
                pms_df.to_csv(os.path.join(gw_folder, 'playermatchstats.csv'), index=False)
            except Exception as e:
                log(f"⚠️  GW{gw_int}: failed to copy playermatchstats - {e}")

        prev_df = gw_slice  # For next delta computation

    log(f"✅ Backfill complete. Wrote {rows_written} discrete player rows across {len(gameweeks)} gameweeks.")
    log(f"Output root: {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
