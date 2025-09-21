"""Utility helpers for operating on list-based tabular datasets."""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

Record = Dict[str, Any]
Dataset = List[Record]


def clone_records(records: Iterable[Mapping[str, Any]]) -> Dataset:
    """Return a list of shallow copies of the provided mapping records."""

    return [dict(row) for row in records]


def sort_records(records: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> Dataset:
    """Return a sorted list of copied records ordered by the provided keys."""

    return sorted(clone_records(records), key=lambda row: tuple(row[key] for key in keys))


def group_indices(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[int]]:
    """Map each group key to the list indices belonging to that group."""

    groups: Dict[Tuple[Any, ...], List[int]] = {}
    for index, row in enumerate(records):
        try:
            key = tuple(row[key] for key in keys)
        except KeyError:
            continue
        groups.setdefault(key, []).append(index)
    return groups


def to_float(value: Any) -> float:
    """Convert arbitrary values to ``float`` while tolerating invalid entries."""

    if value is None:
        return float("nan")
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return float("nan")
        try:
            return float(stripped)
        except ValueError:
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_int(value: Any) -> int | None:
    """Convert a value to ``int`` if possible; otherwise return ``None``."""

    if value is None:
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_nan(value: Any) -> bool:
    """Return ``True`` if the value should be considered missing."""

    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def extract_matrix(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> List[List[float]]:
    """Return a numeric matrix for the requested columns."""

    matrix: List[List[float]] = []
    for row in records:
        matrix.append([to_float(row.get(column)) for column in columns])
    return matrix


def column_values(records: Sequence[Mapping[str, Any]], column: str) -> List[Any]:
    """Collect the values for ``column`` from each record."""

    return [row.get(column) for row in records]


def assign_column(records: Sequence[MutableMapping[str, Any]], column: str, values: Sequence[Any]) -> None:
    """Assign the provided values to ``column`` across the dataset."""

    for row, value in zip(records, values):
        row[column] = value


def filter_records(records: Iterable[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> Dataset:
    """Return records satisfying ``predicate``."""

    return [dict(row) for row in records if predicate(row)]
