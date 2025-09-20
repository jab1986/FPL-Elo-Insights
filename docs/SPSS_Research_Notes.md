## SPSS Research Notes – FPL-Elo-Insights

This document explains how to create and use the analytical dataset `merged_gw.csv` for statistical modeling (e.g., SPSS, R, Python). It operationalizes the research workflow you outlined: per Player–Gameweek rows enriched with opponent + Elo context.

### 1. Building the Dataset
Run (PowerShell):
```
python scripts/build_merged_gw.py
```
Optional: target a different season
```
$env:MERGE_SEASON="2025-2026"; python scripts/build_merged_gw.py
```
Output: `data/<season>/merged_gw.csv`

### 2. Source Inputs
The script consumes existing export pipeline outputs (produced by `scripts/export_data.py`):
* `data/<season>/players.csv` – player identity + position.
* `data/<season>/By Gameweek/GW*/player_gameweek_stats.csv` – per-GW deltas (already solved cumulative subtraction).
* `data/<season>/By Gameweek/GW*/matches.csv` – match context & Elo.
* (Indirect) `playerstats.csv` slices per GW for team inference if a `team` / `team_id` / `team_code` column is present.

### 3. Column Groups in `merged_gw.csv`
Identifier & context:
* `id`, `web_name`, `position`, `gameweek`
* `team_id`, `opponent_team_id`, `is_home`
* `team_elo`, `opponent_elo`, `team_score`, `opponent_score`, `result` (W/D/L), `match_id`

Performance (per-gameweek deltas): all columns from `player_gameweek_stats.csv`, e.g.:
* Minutes / contribution: `minutes`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `saves`, `bonus`, `bps`
* Expected metrics: `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`
* Ownership / pricing snapshots: `now_cost`, `selected_by_percent`, `form`, `event_points`, `value_form`, `value_season`, etc.

Note: Snapshot attributes reflect deadline state, not deltas. Delta-calculated numeric columns are already adjusted vs prior GW in the export pipeline.

### 4. Team Mapping Caveat
`player_gameweek_stats.csv` does not carry a team id. The script attempts to infer `team_id` from each GW's `playerstats.csv` if that file contains one of: `team`, `team_id`, `team_code`. If absent, the match context columns remain blank. (If that occurs in future schema changes, extend the export to persist team id in the discrete file.)

### 5. Using in SPSS
1. Import: `File > Import Data > CSV` → select `merged_gw.csv`.
2. Set measurement levels:
   * Scale: all numeric performance + Elo.
   * Nominal: `position`, `result`.
   * Boolean: convert `is_home` (truth values) to 0/1 if desired.
3. Recommended derived fields (in SPSS or prior in Python):
   * Goal involvement delta: `goals_scored + assists`
   * Net xGI overperformance: `(goals_scored + assists) - expected_goal_involvements`
   * Opponent Elo differential: `team_elo - opponent_elo`
4. Train / validate: use earlier seasons' `merged_gw.csv` as training; current season for out-of-sample validation.

### 6. Extending the Dataset
Add cumulative columns (season-to-date): implement a grouped cumulative sum over deltas by `id` after building, prefix with `cum_`. Example (Python/pandas):
```python
df = pd.read_csv("data/2025-2026/merged_gw.csv")
delta_cols = ["minutes","goals_scored","assists","clean_sheets","goals_conceded","saves","bonus","bps"]
for c in delta_cols:
    df[f"cum_{c}"] = df.groupby("id")[c].cumsum()
df.to_csv("data/2025-2026/merged_gw_with_cum.csv", index=False)
```

### 7. Quality Checks Before Modeling
* No duplicate (`id`,`gameweek`) pairs? (Should be 1-row; investigate if multiples appear.)
* Negative deltas: expected only if source cumulative dropped (should not happen; flag if present outside baseline anomalies).
* Null `team_id` share: if high, confirm presence of team column in GW `playerstats.csv`.

### 8. Repro Workflow Summary
1. Run `python scripts/export_data.py` (updates raw + discrete stats)
2. Run `python scripts/build_merged_gw.py`
3. Import `merged_gw.csv` into SPSS / analytics stack
4. (Optional) Append cumulative features

### 9. Future Enhancements (Optional)
* Add direct team id into `player_gameweek_stats.csv` inside export phase to avoid inference.
* Integrate tournament-specific context by repeating merge for `By Tournament` trees.
* Create a light validation script to assert monotonic cumulative columns pre-delta.

---
Questions or adjustments needed? Open an issue or update this doc alongside schema changes.
