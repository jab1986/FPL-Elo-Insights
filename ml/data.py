from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from .config import PipelineConfig
from .utils import Dataset, clone_records, sort_records, to_float, to_int


def _load_csv(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _coerce_gameweek(value: object) -> int | None:
    numeric = to_float(value)
    if numeric != numeric:  # NaN check
        return None
    return int(numeric)


def load_merged_gameweek_data(config: PipelineConfig) -> Dataset:
    """Load merged gameweek data as a list of dictionaries."""

    records: List[dict] = []
    for season in config.seasons_to_use():
        csv_path = Path(config.base_path) / season / "merged_gw.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing merged_gw.csv for season '{season}' at {csv_path}")

        season_rows = []
        for row in _load_csv(csv_path):
            row = dict(row)
            row["season"] = season

            if "gameweek" not in row:
                raise ValueError(f"File {csv_path} does not contain a 'gameweek' column")

            gameweek = _coerce_gameweek(row.get("gameweek"))
            if gameweek is None:
                continue

            row["gameweek"] = gameweek

            player_id = row.pop("id", row.get("player_id"))
            player_id_int = to_int(player_id)
            if player_id_int is None:
                continue
            row["player_id"] = player_id_int

            min_gw = config.min_gameweek or 1
            if not config.include_preseason and gameweek < min_gw:
                continue
            if config.include_preseason and config.min_gameweek and gameweek < min_gw:
                continue
            if config.max_gameweek is not None and gameweek > config.max_gameweek:
                continue

            season_rows.append(row)

        if not season_rows:
            continue

        records.extend(sort_records(season_rows, ["season", "player_id", "gameweek"]))

    if not records:
        raise ValueError("No data frames were loaded; check the configuration")

    return clone_records(records)
