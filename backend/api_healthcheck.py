"""Simple CLI utility to check API endpoints for the FPL Insights backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


@dataclass
class EndpointResult:
    url: str
    ok: bool
    error: str | None = None
    payload_preview: str | None = None


def check_endpoint(url: str, timeout: int = 10) -> EndpointResult:
    """Fetch a URL and capture a short preview of the response."""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode()
            preview = body[:200]
            try:
                json.loads(body)
            except json.JSONDecodeError:
                pass
            return EndpointResult(url=url, ok=True, payload_preview=preview)
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        return EndpointResult(url=url, ok=False, error=str(exc))


def run_healthcheck(base_url: str, endpoints: Iterable[str]) -> list[EndpointResult]:
    """Run the health check for the provided endpoints."""

    results = []
    for endpoint in endpoints:
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        results.append(check_endpoint(url))
    return results


def main() -> None:  # pragma: no cover - convenience script
    base_url = "http://localhost:8001"
    endpoints = ["/", "/health", "/api/dashboard/stats"]

    print("Testing FPL Insights API endpoints...")
    print("=" * 50)
    for result in run_healthcheck(base_url, endpoints):
        if result.ok:
            print(f"✅ {result.url}: {result.payload_preview}")
        else:
            print(f"❌ {result.url}: {result.error}")
    print("=" * 50)
    print("API testing complete!")


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    main()

