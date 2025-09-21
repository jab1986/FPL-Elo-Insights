"""Tests for the populate_database utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import sys
import types
from typing import Any, Dict, List, Optional

import pytest


if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _AsyncClient:
        async def __aenter__(self) -> "_AsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("HTTP client stub should not issue requests in tests")

    httpx_stub.AsyncClient = _AsyncClient
    sys.modules["httpx"] = httpx_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")

    def load_dotenv(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    dotenv_stub.load_dotenv = load_dotenv
    sys.modules["dotenv"] = dotenv_stub

if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")

    class Client:  # type: ignore[override]
        """Placeholder for the supabase Client type annotation."""

    def create_client(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        raise RuntimeError("Supabase connections are not available in tests")

    supabase_stub.Client = Client
    supabase_stub.create_client = create_client
    sys.modules["supabase"] = supabase_stub

from backend.populate_database import FPLDataPopulator


@dataclass
class FakeResponse:
    """Simple response object mimicking Supabase's return type."""

    data: List[Dict[str, Any]]


class FakeTableQuery:
    """Minimal query builder to emulate the Supabase Python client's behaviour."""

    def __init__(self, client: "FakeSupabaseClient", table_name: str) -> None:
        self._client = client
        self._table_name = table_name
        self._operation: Optional[str] = None
        self._pending_rows: Optional[List[Dict[str, Any]]] = None
        self._filter: Optional[tuple[str, str, Any]] = None

    def delete(self) -> "FakeTableQuery":
        self._operation = "delete"
        self._filter = None
        return self

    def neq(self, column: str, value: Any) -> "FakeTableQuery":
        if column not in self._client.table_columns[self._table_name]:
            raise ValueError(f"column '{column}' does not exist on table '{self._table_name}'")

        self._filter = ("neq", column, value)
        return self

    def insert(self, rows: List[Dict[str, Any]]) -> "FakeTableQuery":
        self._operation = "insert"
        self._pending_rows = rows
        return self

    def execute(self) -> FakeResponse:
        if self._operation == "delete":
            self._client.delete_calls[self._table_name].append(self._filter)
            if self._filter is None:
                self._client.tables[self._table_name] = []
            else:
                _, column, value = self._filter
                self._client.tables[self._table_name] = [
                    row for row in self._client.tables[self._table_name]
                    if row.get(column) == value
                ]
            return FakeResponse(data=[])

        if self._operation == "insert":
            rows = self._pending_rows or []
            unique_key = self._client.unique_keys.get(self._table_name)

            for row in rows:
                if unique_key and any(
                    existing.get(unique_key) == row.get(unique_key)
                    for existing in self._client.tables[self._table_name]
                ):
                    raise ValueError(
                        f"duplicate key value violates unique constraint on {self._table_name}.{unique_key}"
                    )

                self._client.tables[self._table_name].append(row)

            self._client.last_insert_batch[self._table_name] = rows
            return FakeResponse(data=rows)

        raise ValueError("No operation specified before execute() call")


class FakeSupabaseClient:
    """A small stub for exercising the populator without real Supabase access."""

    def __init__(self) -> None:
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            "playermatchstats": [{"id": 1}],
            "matches": [{"id": 1}],
            "players": [{"id": 1}],
            "teams": [{"id": 1}],
            "gameweek_summaries": [{"gameweek": 1}],
        }
        self.table_columns: Dict[str, set[str]] = {
            "playermatchstats": {"id"},
            "matches": {"id"},
            "players": {"id"},
            "teams": {"id"},
            "gameweek_summaries": {"gameweek"},
        }
        self.unique_keys: Dict[str, str] = {"gameweek_summaries": "gameweek"}
        self.delete_calls: Dict[str, List[Optional[tuple[str, str, Any]]]] = defaultdict(list)
        self.last_insert_batch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def table(self, table_name: str) -> FakeTableQuery:
        if table_name not in self.tables:
            raise KeyError(f"Table {table_name} not configured in fake client")
        return FakeTableQuery(self, table_name)


def build_populator(fake_client: FakeSupabaseClient) -> FPLDataPopulator:
    """Create a populator instance backed by the fake Supabase client."""

    populator = FPLDataPopulator.__new__(FPLDataPopulator)
    populator.supabase = fake_client
    return populator


def test_clear_and_populate_gameweek_summaries() -> None:
    fake_client = FakeSupabaseClient()
    populator = build_populator(fake_client)

    populator.clear_existing_data()

    assert fake_client.tables["gameweek_summaries"] == []

    for table in ("playermatchstats", "matches", "players", "teams"):
        assert fake_client.delete_calls[table] == [("neq", "id", 0)]

    assert fake_client.delete_calls["gameweek_summaries"] == [None]

    sample_gameweeks = [
        {
            "id": 1,
            "average_entry_score": 50,
            "highest_score": 120,
            "deadline_time": "2024-08-01T18:30:00Z",
            "deadline_time_epoch": 1_724_825_400,
            "finished": True,
            "data_checked": True,
            "highest_scoring_entry": 123,
            "most_selected": 500,
            "most_transferred_in": 501,
            "top_element": 10,
            "top_element_info": {"id": 10, "points": 15},
            "most_captained": 11,
            "most_vice_captained": 12,
        }
    ]

    populator.populate_gameweeks(sample_gameweeks)

    assert fake_client.tables["gameweek_summaries"] == [
        {
            "gameweek": 1,
            "average_entry_score": 50.0,
            "highest_score": 120,
            "deadline_time": "2024-08-01T18:30:00Z",
            "deadline_time_epoch": 1_724_825_400,
            "finished": True,
            "data_checked": True,
            "highest_scoring_entry": 123,
            "most_selected": 500,
            "most_transferred_in": 501,
            "top_element": 10,
            "top_element_info": {"id": 10, "points": 15},
            "most_captained": 11,
            "most_vice_captained": 12,
        }
    ]

    with pytest.raises(ValueError):
        populator.populate_gameweeks(sample_gameweeks)
