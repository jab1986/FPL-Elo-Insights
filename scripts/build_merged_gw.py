"""Build a consolidated per-player per-gameweek dataset (merged_gw.csv).

This script constructs a long-form analytical table combining:
  - Per-GW player deltas from `player_gameweek_stats.csv`
  - Player identity & position from season `players.csv`
  - Match context from each GW `matches.csv` (home/away, opponent id, result)
  - Team & opponent Elo from `matches.csv` (home_team_elo / away_team_elo)

Output: data/{SEASON}/merged_gw.csv

Assumptions / Conventions:
  - Season directory already populated by export_data.py
  - Each GW folder under `By Gameweek/GW{n}` contains: matches.csv, player_gameweek_stats.csv
  - Player unique key alignment: player_gameweek_stats.id == players.player_id
  - Team mapping: matches.{home_team,away_team} correspond to teams.id (int)
  - Player to team mapping uses latest `playerstats` per GW slice if available for team inference fallback isn't needed for deltas (player_gameweek_stats does not include team id).

Enhancements (future): add cumulative season-to-date columns (prefix cum_) by re-accumulating deltas; integrate tournament-sliced context.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
from typing import List, Dict

SEASON = os.environ.get("MERGE_SEASON", "2025-2026")
BASE_PATH = os.path.join("data", SEASON)
BY_GW_PATH = os.path.join(BASE_PATH, "By Gameweek")
OUTPUT_CSV = os.path.join(BASE_PATH, "merged_gw.csv")
STATIC_MAPPING_PATH = os.path.join(BASE_PATH, "player_team_mapping.csv")


def _log(msg: str):
    print(msg, flush=True)


def load_players() -> pd.DataFrame:
    p_path = os.path.join(BASE_PATH, "players.csv")
    if not os.path.exists(p_path):
        # Legacy (2024-2025) stores players under players/players.csv
        legacy_p_path = os.path.join(BASE_PATH, "players", "players.csv")
        if os.path.exists(legacy_p_path):
            p_path = legacy_p_path
            _log(f"ℹ️  Using legacy players file at {p_path}")
        else:
            _log(f"❌ players.csv not found at {p_path}")
            sys.exit(1)
    df = pd.read_csv(p_path)
    needed = ["player_id", "web_name", "position"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        _log(f"❌ Missing columns in players.csv: {missing}")
        sys.exit(1)
    # Include team_code if present for modern season mapping
    cols = needed + (["team_code"] if "team_code" in df.columns else [])
    return df[cols].rename(columns={"player_id": "id"})


def build_player_team_mapping(gameweeks: List[int]) -> pd.DataFrame:
    """Build a (id, gameweek) -> team_id mapping using playermatchstats + matches.

    Strategy:
      - For each GW, read playermatchstats.csv if present.
      - Identify potential player id column: prefer 'player_id', fallback 'id'.
      - Identify team indicator: look for 'team_id'/'team'. If absent, derive from match involvement:
          * Join on match_id with matches.csv to get home_team / away_team.
          * If the player's side indicator columns exist (e.g., 'was_home') map to home/away.
          * Fallback: if a column 'is_home' present treat similarly.
      - Keep first occurrence per player per GW (players can appear only once meaningfully per GW for mapping).
      - Return DataFrame with columns ['id','gameweek','team_id'].
    """
    records = []
    for gw in gameweeks:
        pms_path = os.path.join(BY_GW_PATH, f"GW{gw}", "playermatchstats.csv")
        if not os.path.exists(pms_path):
            continue
        try:
            pms = pd.read_csv(pms_path)
        except Exception as e:
            _log(f"⚠️  GW{gw}: failed to read playermatchstats.csv - {e}")
            continue

        # Determine player id column
        pid_col = None
        for cand in ["player_id", "id"]:
            if cand in pms.columns:
                pid_col = cand
                break
        if pid_col is None:
            continue

        # Direct team column present?
        direct_team_col = None
        for cand in ["team_id", "team", "team_code"]:
            if cand in pms.columns:
                direct_team_col = cand
                break

        if direct_team_col is not None:
            mini = pms[[pid_col, direct_team_col]].dropna().drop_duplicates().copy()
            if not mini.empty:
                mini.rename(columns={pid_col: "id", direct_team_col: "team_id"}, inplace=True)
                mini["gameweek"] = gw
                records.append(mini[["id", "team_id", "gameweek"]])
            continue

        # Need to derive via matches
        matches_path = os.path.join(BY_GW_PATH, f"GW{gw}", "matches.csv")
        if not os.path.exists(matches_path):
            continue
        try:
            matches = pd.read_csv(matches_path)
        except Exception as e:
            _log(f"⚠️  GW{gw}: failed to read matches.csv for team derivation - {e}")
            continue

        needed_cols = {"match_id", "home_team", "away_team"}
        if not needed_cols.issubset(matches.columns):
            # Cannot derive
            continue

        # Join pms to matches on match_id
        if "match_id" not in pms.columns:
            continue

        joined = pms.merge(matches[["match_id", "home_team", "away_team"]], on="match_id", how="left")

        # Determine side indicator
        side_col = None
        for cand in ["was_home", "is_home", "home"]:
            if cand in joined.columns:
                side_col = cand
                break

        if side_col is not None:
            # Normalize boolean / 0/1 values
            def pick_team(row):
                val = row.get(side_col)
                if pd.isna(val):
                    return None
                try:
                    if isinstance(val, str):
                        lv = val.lower()
                        if lv in {"true", "t", "1", "yes", "y"}:
                            return row.get("home_team")
                        if lv in {"false", "f", "0", "no", "n"}:
                            return row.get("away_team")
                    # numeric
                    if float(val) == 1:
                        return row.get("home_team")
                    if float(val) == 0:
                        return row.get("away_team")
                except Exception:
                    pass
                return None

            joined["team_id"] = joined.apply(pick_team, axis=1)
        else:
            # If no side column, attempt heuristic: if exactly one of player_id appears in rows whose match row home/away teams differ (?) -> ambiguous, so skip.
            # Without side info we cannot safely infer.
            continue

        mini = joined[[pid_col, "team_id"]].dropna().drop_duplicates()
        if mini.empty:
            continue
        mini.rename(columns={pid_col: "id"}, inplace=True)
        mini["gameweek"] = gw
        records.append(mini[["id", "team_id", "gameweek"]])

    if not records:
        _log("⚠️  Could not build any player->team mappings from playermatchstats.")
        return pd.DataFrame(columns=["id", "team_id", "gameweek"])

    mapping = pd.concat(records, ignore_index=True)
    # Deduplicate preferring first occurrence (already ensured by drop_duplicates order)
    mapping = mapping.drop_duplicates(subset=["id", "gameweek"], keep="first")
    return mapping


def list_gameweeks() -> List[int]:
    if not os.path.isdir(BY_GW_PATH):
        _log(f"❌ By Gameweek path not found: {BY_GW_PATH}")
        sys.exit(1)
    gws = []
    for d in os.listdir(BY_GW_PATH):
        if d.startswith("GW"):
            try:
                gws.append(int(d[2:]))
            except ValueError:
                continue
    return sorted(gws)


def load_gw_stats(gw: int) -> pd.DataFrame | None:
    path = os.path.join(BY_GW_PATH, f"GW{gw}", "player_gameweek_stats.csv")
    if not os.path.exists(path):
        _log(f"⚠️  GW{gw}: player_gameweek_stats.csv missing – skipping.")
        return None
    df = pd.read_csv(path)
    df["gameweek"] = gw
    return df


def load_gw_matches(gw: int) -> pd.DataFrame | None:
    path = os.path.join(BY_GW_PATH, f"GW{gw}", "matches.csv")
    if not os.path.exists(path):
        _log(f"⚠️  GW{gw}: matches.csv missing – skipping match context.")
        return None
    df = pd.read_csv(path)
    # Ensure required fields exist; if not, create placeholders
    essentials = [
        "gameweek", "home_team", "away_team", "home_team_elo", "away_team_elo",
        "home_score", "away_score", "match_id"
    ]
    for col in essentials:
        if col not in df.columns:
            df[col] = None
    return df


def enrich_with_match_context(stats_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Expand per-player rows with opponent, is_home, team_elo, opponent_elo, result.

    Note: player_gameweek_stats does not include team id. We infer team via participation by joining
    playermatchstats if needed (future). For now we approximate using minutes delta from stats_df + match rosters is not available, so we cannot perfectly map team for bench/no-minutes players.

    Interim heuristic: derive team assignment via sign of influence from cumulative? Not available.

    => Simplification: If we cannot confidently derive team, we will skip team-level expansion.
       To retain requested columns, we'll attempt a fallback merge using playerstats cumulative slice per GW.
    """
    # At this point we expect a 'team_id' column already merged (added earlier in main via team mapping step).
    if 'team_id' not in stats_df.columns:
        _log("⚠️  No team_id present on stats rows; skipping match enrichment.")
        return stats_df
    # Build a long match context table: one row per (gameweek, team perspective)
    # We'll duplicate match rows for home and away to join on team_id.
    ctx_home = matches_df[[
        "gameweek", "match_id", "home_team", "away_team", "home_team_elo", "away_team_elo", "home_score", "away_score"
    ]].copy()
    ctx_home.rename(columns={
        "home_team": "team_id", "away_team": "opponent_team_id",
        "home_team_elo": "team_elo", "away_team_elo": "opponent_elo",
        "home_score": "team_score", "away_score": "opponent_score"
    }, inplace=True)
    ctx_home["is_home"] = True

    ctx_away = matches_df[[
        "gameweek", "match_id", "home_team", "away_team", "home_team_elo", "away_team_elo", "home_score", "away_score"
    ]].copy()
    ctx_away.rename(columns={
        "away_team": "team_id", "home_team": "opponent_team_id",
        "away_team_elo": "team_elo", "home_team_elo": "opponent_elo",
        "away_score": "team_score", "home_score": "opponent_score"
    }, inplace=True)
    ctx_away["is_home"] = False

    match_long = pd.concat([ctx_home, ctx_away], ignore_index=True)

    enriched = stats_df.merge(
        match_long,
        on=["gameweek", "team_id"],
        how="left"
    )

    # Result classification
    def classify(row):
        if pd.isna(row.get("team_score")) or pd.isna(row.get("opponent_score")):
            return None
        if row["team_score"] > row["opponent_score"]:
            return "W"
        if row["team_score"] < row["opponent_score"]:
            return "L"
        return "D"

    enriched["result"] = enriched.apply(classify, axis=1)
    return enriched


def main():
    _log(f"--- Building merged_gw for season {SEASON} ---")

    players = load_players()
    gws = list_gameweeks()
    if not gws:
        _log("No gameweeks found – aborting.")
        sys.exit(1)

    all_stats = []
    all_matches = []
    for gw in gws:
        stats_df = load_gw_stats(gw)
        m_df = load_gw_matches(gw)
        if stats_df is None or m_df is None:
            continue
        all_stats.append(stats_df)
        all_matches.append(m_df)

    if not all_stats:
        _log("❌ No player_gameweek_stats discovered; nothing to merge.")
        sys.exit(1)

    stats_all = pd.concat(all_stats, ignore_index=True)
    matches_all = pd.concat(all_matches, ignore_index=True)

    # Add player identifiers & position
    stats_all = stats_all.merge(players, on="id", how="left")

    # Static mapping first (for legacy seasons)
    static_map = None
    if os.path.exists(STATIC_MAPPING_PATH):
        try:
            sm = pd.read_csv(STATIC_MAPPING_PATH)
            if {"id", "team_id"}.issubset(sm.columns):
                static_map = sm[["id", "team_id"]].drop_duplicates()
                _log(f"ℹ️  Loaded static team mapping: {len(static_map)} players")
            else:
                _log("⚠️  Static mapping file missing required columns 'id','team_id'. Ignoring.")
        except Exception as e:
            _log(f"⚠️  Failed reading static mapping file: {e}")

    if static_map is not None:
        # Broadcast static mapping to all gameweeks
        gw_frame = pd.DataFrame({"gameweek": gws})
        static_expanded = gw_frame.assign(key=1).merge(static_map.assign(key=1), on="key").drop(columns="key")
        stats_all = stats_all.merge(static_expanded, on=["id", "gameweek"], how="left")
    else:
        # Try direct players.team_code (modern season convenience)
        if 'team_code' in stats_all.columns:
            # treat team_code as stable team identifier
            stats_all.rename(columns={'team_code': 'team_id'}, inplace=True)
            _log("ℹ️  Using players.team_code as team_id mapping.")
        else:
            # Build player->team mapping from playermatchstats
            team_map = build_player_team_mapping(gws)
            if not team_map.empty:
                stats_all = stats_all.merge(team_map, on=["id", "gameweek"], how="left")
            else:
                _log("⚠️  Team mapping empty; match context will be skipped.")

    # Reorder: identifiers first
    id_cols = [c for c in ["id", "web_name", "position", "gameweek"] if c in stats_all.columns]
    other_cols = [c for c in stats_all.columns if c not in id_cols]
    stats_all = stats_all[id_cols + other_cols]

    enriched = enrich_with_match_context(stats_all, matches_all)

    # Provide a concise column ordering placing match context after identifiers
    context_cols = [
        "team_id", "opponent_team_id", "is_home", "team_elo", "opponent_elo", "team_score", "opponent_score", "result", "match_id"
    ]
    ordered_context = [c for c in context_cols if c in enriched.columns]
    final_cols = id_cols + ordered_context + [c for c in enriched.columns if c not in id_cols + ordered_context]
    enriched = enriched[final_cols]

    enriched.to_csv(OUTPUT_CSV, index=False)
    _log(f"✅ Wrote {len(enriched)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
