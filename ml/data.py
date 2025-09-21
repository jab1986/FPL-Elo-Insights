"""Data loading helpers for ML pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .config import PipelineConfig


def load_merged_gameweek_data(config: PipelineConfig) -> pd.DataFrame:
    """Load merged gameweek data for the configured seasons."""

    frames: List[pd.DataFrame] = []
    for season in config.seasons_to_use():
        csv_path = Path(config.base_path) / season / "merged_gw.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing merged_gw.csv for season '{season}' at {csv_path}")

        frame = pd.read_csv(csv_path)
        frame["season"] = season

        if "gameweek" not in frame.columns:
            raise ValueError(f"File {csv_path} does not contain a 'gameweek' column")

        frame["gameweek"] = pd.to_numeric(frame["gameweek"], errors="coerce")
        frame = frame.dropna(subset=["gameweek"]).copy()
        frame["gameweek"] = frame["gameweek"].astype(int)

        if not config.include_preseason:
            frame = frame[frame["gameweek"] >= config.min_gameweek]
        elif config.min_gameweek:
            frame = frame[frame["gameweek"] >= config.min_gameweek]

        if config.max_gameweek is not None:
            frame = frame[frame["gameweek"] <= config.max_gameweek]

        frame.rename(columns={"id": "player_id"}, inplace=True)
        frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["player_id"])

        frames.append(frame.reset_index(drop=True))

    if not frames:
        raise ValueError("No data frames were loaded; check the configuration")

    combined = pd.concat(frames, ignore_index=True)
    return combined

