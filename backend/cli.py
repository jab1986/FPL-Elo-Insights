"""Command line interface for interacting with FPL Insights data."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from typing import Any, Callable, Dict, Optional

import httpx
import typer
from dotenv import load_dotenv

# Load environment variables first before importing services
load_dotenv()

from app.services import mock_data
from app.services.data_service import data_service
from app.services.fpl_service import fpl_service

app = typer.Typer(help="FPL Insights command line interface")
players_app = typer.Typer(help="Player related commands")
teams_app = typer.Typer(help="Team related commands")
matches_app = typer.Typer(help="Match related commands")
match_stats_app = typer.Typer(help="Player match statistics commands")
gameweeks_app = typer.Typer(help="Gameweek summary commands")
dashboard_app = typer.Typer(help="Dashboard overview commands")
user_team_app = typer.Typer(help="Fantasy Premier League entry commands")


app.add_typer(players_app, name="players")
app.add_typer(teams_app, name="teams")
app.add_typer(matches_app, name="matches")
app.add_typer(match_stats_app, name="player-match-stats")
app.add_typer(gameweeks_app, name="gameweeks")
app.add_typer(dashboard_app, name="dashboard")
app.add_typer(user_team_app, name="user-team")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _print_result(result: Any) -> None:
    typer.echo(json.dumps(result, indent=2, default=_json_default))


def _warn(message: str) -> None:
    typer.secho(message, fg=typer.colors.YELLOW)


def _error(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED)


def _fetch_with_fallback(
    fetcher: Callable[[], Any], fallback: Callable[[], Any], dataset: str
) -> Any:
    try:
        return fetcher()
    except Exception as exc:  # pragma: no cover - defensive fallback
        _warn(
            f"Unable to retrieve {dataset} from the live service ({exc}). Using sample data."
        )
        return fallback()


@players_app.command("list")
def list_players(
    position: Optional[str] = typer.Option(
        None, "--position", "-p", help="Filter by player position"
    ),
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Filter by team name"
    ),
    min_price: Optional[int] = typer.Option(
        None,
        "--min-price",
        help="Minimum price filter (use FPL tenths, e.g. 75 for £7.5m)",
    ),
    max_price: Optional[int] = typer.Option(
        None,
        "--max-price",
        help="Maximum price filter (use FPL tenths, e.g. 120 for £12.0m)",
    ),
    min_points: Optional[int] = typer.Option(
        None, "--min-points", help="Minimum total points filter"
    ),
    max_points: Optional[int] = typer.Option(
        None, "--max-points", help="Maximum total points filter"
    ),
    gameweek: Optional[int] = typer.Option(
        None, "--gameweek", "-g", help="Filter by gameweek"
    ),
) -> None:
    """List players using optional filters."""

    filters: Dict[str, Any] = {}
    if position:
        filters["position"] = position
    if team:
        filters["team"] = team
    if min_price is not None:
        filters["min_price"] = min_price
    if max_price is not None:
        filters["max_price"] = max_price
    if min_points is not None:
        filters["min_points"] = min_points
    if max_points is not None:
        filters["max_points"] = max_points
    if gameweek is not None:
        filters["gameweek"] = gameweek

    players = _fetch_with_fallback(
        lambda: data_service.get_players(filters),
        lambda: mock_data.sample_players(filters),
        "players",
    )
    _print_result(players)


@players_app.command("get")
def get_player(player_id: int = typer.Argument(..., help="Player identifier")) -> None:
    """Display a single player's information."""

    try:
        player = data_service.get_player_by_id(player_id)
    except Exception as exc:  # pragma: no cover - defensive fallback
        _warn(
            f"Unable to retrieve player {player_id} from the live service ({exc}). Using sample data."
        )
        player = mock_data.sample_player(player_id)

    if not player:
        _error(f"Player {player_id} not found.")
        raise typer.Exit(code=1)

    _print_result(player)


@players_app.command("top")
def top_players(
    limit: int = typer.Option(10, "--limit", "-l", min=1, help="Number of players to return")
) -> None:
    """Display top performing players."""

    players = _fetch_with_fallback(
        lambda: data_service.get_top_players(limit),
        lambda: mock_data.sample_top_players(limit),
        "top players",
    )
    _print_result(players)


@players_app.command("search")
def search_players(
    query: str = typer.Argument(..., help="Search query for player names")
) -> None:
    """Search players by name."""

    if len(query.strip()) < 3:
        _warn("Search queries work best with at least 3 characters.")

    players = _fetch_with_fallback(
        lambda: data_service.search_players(query),
        lambda: mock_data.sample_search_players(query),
        "players",
    )
    _print_result(players)


@teams_app.command("list")
def list_teams() -> None:
    """List all teams."""

    teams = _fetch_with_fallback(
        data_service.get_teams,
        mock_data.sample_teams,
        "teams",
    )
    _print_result(teams)


@teams_app.command("get")
def get_team(team_id: int = typer.Argument(..., help="Team identifier")) -> None:
    """Display a single team's information."""

    try:
        team = data_service.get_team_by_id(team_id)
    except Exception as exc:  # pragma: no cover - defensive fallback
        _warn(
            f"Unable to retrieve team {team_id} from the live service ({exc}). Using sample data."
        )
        team = mock_data.sample_team(team_id)

    if not team:
        _error(f"Team {team_id} not found.")
        raise typer.Exit(code=1)

    _print_result(team)


@teams_app.command("search")
def search_teams(
    query: str = typer.Argument(..., help="Search query for team names")
) -> None:
    """Search teams by name."""

    if len(query.strip()) < 2:
        _warn("Search queries work best with at least 2 characters.")

    teams = _fetch_with_fallback(
        lambda: data_service.search_teams(query),
        lambda: mock_data.sample_search_teams(query),
        "teams",
    )
    _print_result(teams)


@matches_app.command("list")
def list_matches(
    gameweek: Optional[int] = typer.Option(
        None, "--gameweek", "-g", help="Filter matches by gameweek"
    )
) -> None:
    """List fixtures and results."""

    matches = _fetch_with_fallback(
        lambda: data_service.get_matches(gameweek),
        lambda: mock_data.sample_matches(gameweek),
        "matches",
    )
    _print_result(matches)


@matches_app.command("get")
def get_match(match_id: int = typer.Argument(..., help="Match identifier")) -> None:
    """Display a single match's information."""

    try:
        match = data_service.get_match_by_id(match_id)
    except Exception as exc:  # pragma: no cover - defensive fallback
        _warn(
            f"Unable to retrieve match {match_id} from the live service ({exc}). Using sample data."
        )
        match = mock_data.sample_match(match_id)

    if not match:
        _error(f"Match {match_id} not found.")
        raise typer.Exit(code=1)

    _print_result(match)


@match_stats_app.command("list")
def list_match_stats(
    match_id: Optional[int] = typer.Option(None, "--match-id", help="Filter by match ID"),
    player_id: Optional[int] = typer.Option(None, "--player-id", help="Filter by player ID"),
) -> None:
    """List player match statistics."""

    stats = _fetch_with_fallback(
        lambda: data_service.get_player_match_stats(match_id=match_id, player_id=player_id),
        lambda: mock_data.sample_player_match_stats(match_id=match_id, player_id=player_id),
        "player match stats",
    )
    _print_result(stats)


@gameweeks_app.command("list")
def list_gameweeks() -> None:
    """List gameweek summaries."""

    summaries = _fetch_with_fallback(
        data_service.get_gameweek_summaries,
        mock_data.sample_gameweek_summaries,
        "gameweek summaries",
    )
    _print_result(summaries)


@gameweeks_app.command("current")
def current_gameweek() -> None:
    """Show the current or most recent gameweek."""

    try:
        summary = data_service.get_current_gameweek()
    except Exception as exc:  # pragma: no cover - defensive fallback
        _warn(
            f"Unable to determine the current gameweek from the live service ({exc}). Using sample data."
        )
        summary = mock_data.sample_current_gameweek()

    if not summary:
        _error("Gameweek information is unavailable.")
        raise typer.Exit(code=1)

    _print_result(summary)


@dashboard_app.command("stats")
def dashboard_stats() -> None:
    """Display aggregated dashboard statistics."""

    stats = _fetch_with_fallback(
        data_service.get_dashboard_stats,
        mock_data.sample_dashboard_stats,
        "dashboard stats",
    )
    _print_result(stats)


@user_team_app.command("show")
def show_user_team(
    team_id: int = typer.Argument(..., help="Fantasy Premier League entry ID"),
    event: Optional[int] = typer.Option(
        None, "--event", "-e", help="Specific gameweek to retrieve picks for"
    ),
) -> None:
    """Retrieve an FPL entry with enriched player picks."""

    async def _fetch() -> Dict[str, Any]:
        return await fpl_service.get_user_team(team_id, event)

    try:
        result = asyncio.run(_fetch())
    except httpx.HTTPStatusError as exc:  # pragma: no cover - network fallback
        if team_id == 266343:
            _warn(
                "Live FPL service is unavailable or blocked; using bundled sample team data."
            )
            result = fpl_service.get_sample_user_team()
        else:
            _error(
                f"FPL service responded with {exc.response.status_code}: {exc.response.text}"
            )
            raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover - defensive fallback
        if team_id == 266343:
            _warn(
                "Unable to fetch live team data; returning the bundled sample team payload."
            )
            result = fpl_service.get_sample_user_team()
        else:
            _error(f"Unable to fetch team {team_id}: {exc}")
            raise typer.Exit(code=1)

    _print_result(result)


if __name__ == "__main__":
    app()
