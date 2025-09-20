## FPL-Elo-Insights – AI Coding Assistant Guidelines

Purpose: Maintain and evolve a public Fantasy Premier League + Elo dataset. Core value = reliable, reproducible CSV snapshots (season master, per gameweek, per tournament, and derived discrete stats).

### 1. Core Data Model (Anchor Concepts)
Season root (`data/{season}/`): `players.csv`, `teams.csv`, `playerstats.csv` (cumulative per player per GW), `gameweek_summaries.csv`.
By Gameweek (`By Gameweek/GW{n}/`): Snapshot at GW end. Contains: `fixtures.csv`, `matches.csv`, `players.csv`, `teams.csv`, `playerstats.csv`, `playermatchstats.csv`, generated `player_gameweek_stats.csv` (discrete deltas for that GW only).
By Tournament (`By Tournament/{Tournament}/GW{n}/`): Same schema as By Gameweek but filtered to that competition; deltas for tournament GW are still computed against the global previous GW (see export logic).
Legacy season 2024-2025 has earlier organization under `matches/` & `playermatchstats/` plus helper scripts to backfill per-GW folders.

### 2. Pipeline Overview (Modern Season 2025-2026)
Primary script: `scripts/export_data.py` (authoritative). Steps:
1. Fetch full tables from Supabase (environment vars: `SUPABASE_URL`, `SUPABASE_KEY`). Pagination in 1000-row batches.
2. Derive `tournament` slug from `match_id`; filter out friendlies + GW0.
3. Write master season CSVs (always latest state – NOT immutable).
4. Populate `By Tournament` then `By Gameweek` directories. File locking rule: if a GW is finished AND snapshot `players.csv` & `teams.csv` already exist, only dynamic files (`matches.csv`, `fixtures.csv`, `playermatchstats.csv`, `playerstats.csv`) refresh; otherwise write full snapshot.
5. Compute discrete per-GW stats via `calculate_discrete_gameweek_stats()` producing `player_gameweek_stats.csv` in every GW folder (and per tournament) by subtracting previous cumulative values for columns in `CUMULATIVE_COLS`. Baseline GW uses raw values.

### 3. Discrete vs Cumulative Player Data
`playerstats.csv` = cumulative totals (season to that GW). `player_gameweek_stats.csv` = per-GW deltas (preferred for analysis). ALWAYS merge on `id` (player id). Snapshot columns (e.g., `now_cost`, `selected_by_percent`) represent state at GW deadline, not a delta.

### 4. Naming & Tournament Mapping
Tournament slug extraction scans `match_id` for keys in `TOURNAMENT_NAME_MAP` (see `export_data.py`). Slugs mapped to human folder names (e.g., `premier-league` -> `Premier League`). When adding a new competition, extend `TOURNAMENT_NAME_MAP` and ensure `match_id` embeds the slug.

### 5. Historical Integrity Rules
Never retro-edit locked snapshot `players.csv` / `teams.csv` for a finished GW—create a corrective script if needed (document rationale). Dynamic stats files can refresh safely (they reflect evolving completeness until all matches processed). Do not recompute deltas out of order: subtraction assumes contiguous GW numbering.

### 6. Legacy Utility Scripts (2024-2025)
`split_by_gameweek.py` & `fixcsv.py`: One-off normalization / splitting of monolithic CSVs into GW folders using `gameweek` and `match_id` mapping. `split_csv_data.py`: Incremental update logic from latest finished GW onward (skips earlier GWs). For new work prefer enhancing `export_data.py`; touch legacy scripts only for archival maintenance.

### 7. Adding New Derived Metrics
Add to `CUMULATIVE_COLS` ONLY if the metric is season-accumulating. For per-GW state variables, add to `SNAPSHOT_COLS`. Any new identifier fields belong in `ID_COLS`. Keep delta calculation stable: subtraction loop expects the columns exist in both current and previous DataFrames.

### 8. Environment & Execution
Run pipeline locally (Windows PowerShell) with: `python scripts/export_data.py` (requires `supabase-py`, `pandas`). Ensure env vars loaded (use a .env loader or manual export). No build system or tests currently—favor creating lightweight validation notebooks or quick pandas assertions when extending.

### 9. Common Extension Tasks (Examples)
Add competition: update `TOURNAMENT_NAME_MAP`, rerun export (historical backfill may require manual script to reprocess matches filtering logic).
Add new player stat: ensure Supabase table provides it; include in appropriate COLS list; verify presence before selecting subset, guarding with `[col for col in ... if col in df.columns]` pattern (already used—preserve it).
Improve delta reliability: consider caching merged previous GW subset; avoid re-reading large CSVs repeatedly if scaling.

### 10. Data Quality Guardrails (Implement When Contributing Code)
Log counts after filtering tournaments; warn if any required table fetch returns empty. Before writing deltas, assert sorted GW continuity to avoid negative values. If introducing parallelism, preserve ordered dependency for previous GW subtraction.

### 11. DO / AVOID
DO: Reuse existing column selection + safe existence checks. Keep file naming consistent. Document any schema drift in this file.
AVOID: Writing partial snapshots for finished GWs; renaming core CSVs; introducing non-deterministic timestamp columns inside historical snapshots.

### 12. Quick Reference
Key script: `scripts/export_data.py`
Core constants: `CUMULATIVE_COLS`, `SNAPSHOT_COLS`, `ID_COLS`, `TOURNAMENT_NAME_MAP`.
Primary outputs: season master CSVs + GW snapshots + per-tournament filtered snapshots + discrete stats.

Feedback welcome: Clarify unclear sections or request expansion (e.g., Supabase schema, adding tests) before large refactors.
