"""Deprecated module: mock_data

This module has been disabled. The application operates against live data
sources only (Supabase and the public FPL API). Keeping this file in place
prevents import errors in older branches, but any import will fail fast.
"""

raise RuntimeError(
    "backend.app.services.mock_data has been removed. Use live data via DataService/FPLService."
)
