"""Optional third-party dependency helpers for the ML pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Optional


def _safe_import(module_name: str) -> Optional[object]:
    """Attempt to import a module, returning ``None`` on failure."""

    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


pandas = _safe_import("pandas")
numpy = _safe_import("numpy")
_sklearn_linear_model = _safe_import("sklearn.linear_model")

RidgeClass = None
if _sklearn_linear_model is not None:
    RidgeClass = getattr(_sklearn_linear_model, "Ridge", None)

__all__ = ["pandas", "numpy", "RidgeClass"]
