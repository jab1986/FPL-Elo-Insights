"""Configuration objects for the ML prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the player points prediction pipeline."""

    seasons: Tuple[str, ...] = ("2024-2025", "2025-2026")
    """Seasons to include when building the dataset."""

    base_path: Path = Path("data")
    """Root folder containing the raw CSV exports."""

    target: str = "event_points"
    """Target column to predict."""

    min_gameweek: int = 1
    """Earliest gameweek to consider (inclusive)."""

    max_gameweek: Optional[int] = None
    """Maximum gameweek to include. ``None`` keeps all available gameweeks."""

    include_preseason: bool = False
    """Whether to keep preseason friendlies (stored as GW0 in the dataset)."""

    holdout_gameweeks: int = 2
    """Number of most recent gameweeks per season reserved for evaluation."""

    min_history_games: int = 3
    """Minimum number of prior appearances required before a sample is used."""

    event_point_windows: Tuple[int, ...] = (3, 5)
    """Rolling windows (in matches) used for past event point aggregates."""

    expectation_windows: Tuple[int, ...] = (3,)
    """Rolling windows for expected goal metrics and ICT index features."""

    team_form_windows: Tuple[int, ...] = (3,)
    """Rolling windows (per club) for contextual team form features."""

    cross_validation_folds: int = 0
    """Number of rolling-origin validation folds to compute (0 disables CV)."""

    cv_min_train_gameweeks: int = 4
    """Minimum number of historical gameweeks required before a CV evaluation."""

    alpha_grid: Tuple[float, ...] = ()
    """Candidate ridge alphas evaluated during automatic model selection."""

    tuning_metric: str = "rmse"
    """Metric used to score hyperparameter candidates during tuning."""

    tuning_folds: int = 3
    """Number of rolling-origin folds used when evaluating alpha candidates."""

    tuning_min_train_gameweeks: int = 4
    """Minimum history required before producing a tuning evaluation."""

    output_dir: Path = Path("ml/artifacts")
    """Where to persist pipeline outputs such as metrics and coefficients."""

    model_alpha: float = 5.0
    """Regularisation strength for the custom ridge regression model."""

    random_state: Optional[int] = None
    """Placeholder for compatibility with future stochastic models."""

    uncertainty_confidence_levels: Tuple[float, ...] = (0.8, 0.95)
    """Confidence intervals generated around the point predictions."""

    def seasons_to_use(self) -> Tuple[str, ...]:
        """Return the configured seasons as a tuple."""

        return tuple(self.seasons)

    def resolved_output_dir(self) -> Path:
        """Return the output directory, ensuring it is relative to the project root."""

        return Path(self.output_dir)

