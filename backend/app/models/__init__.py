from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


# Player Models
class PlayerBase(BaseModel):
    id: int
    name: str
    team: str
    position: str
    now_cost: int
    selected_by_percent: float
    total_points: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    saves: int
    bonus: int
    bps: int
    influence: float
    creativity: float
    threat: float
    ict_index: float
    starts: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    value_season: float
    value_form: float
    points_per_game: float
    transfers_in: int
    transfers_out: int
    transfers_in_event: int
    transfers_out_event: int


# Team Models
class TeamBase(BaseModel):
    id: int
    name: str
    short_name: str
    strength: int
    played: int
    win: int
    draw: int
    loss: int
    points: int
    position: int
    form: str
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int


# Match Models
class MatchBase(BaseModel):
    id: int
    gameweek: int
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    kickoff_time: datetime
    finished: bool
    minutes: int
    provisional_start_time: bool
    finished_provisional: bool
    started: bool


# Player Match Stats Models
class PlayerMatchStatsBase(BaseModel):
    id: int
    match_id: int
    player_id: int
    team_id: int
    position: str
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    saves: int
    bonus: int
    bps: int
    influence: float
    creativity: float
    threat: float
    ict_index: float
    total_points: int
    in_dreamteam: bool


# Gameweek Summary Models
class GameweekSummaryBase(BaseModel):
    gameweek: int
    average_entry_score: float
    highest_score: int
    deadline_time: datetime
    deadline_time_epoch: int
    finished: bool
    data_checked: bool
    highest_scoring_entry: Optional[int]
    most_selected: Optional[int]
    most_transferred_in: Optional[int]
    top_element: Optional[int]
    top_element_info: Optional[dict]
    most_captained: Optional[int]
    most_vice_captained: Optional[int]


# Dashboard Models
class DashboardStats(BaseModel):
    totalPlayers: int
    totalTeams: int
    currentGameweek: int
    averagePoints: float
    topScorer: Optional[PlayerBase]
    mostValuable: Optional[PlayerBase]


# Chart Data Models
class PerformanceDataPoint(BaseModel):
    gameweek: int
    points: int
    elo: float
    date: str


# API Response Models
class PaginatedResponse(BaseModel):
    data: List[dict]
    total: int
    page: int
    limit: int


# Filter Models
class PlayerFilters(BaseModel):
    position: Optional[str] = None
    team: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    min_points: Optional[int] = None
    max_points: Optional[int] = None
    gameweek: Optional[int] = None


class MatchFilters(BaseModel):
    gameweek: Optional[int] = None
    team: Optional[str] = None
    finished: Optional[bool] = None


# User Team Models
class UserTeamPlayerInfo(BaseModel):
    id: Optional[int]
    web_name: Optional[str]
    first_name: Optional[str]
    second_name: Optional[str]
    team: Optional[str]
    team_short_name: Optional[str]
    position: Optional[str]
    now_cost: Optional[float]
    total_points: Optional[int]
    selected_by_percent: Optional[float]
    event_points: Optional[int] = None
    status: Optional[str] = None


class UserTeamPick(BaseModel):
    element: Optional[int]
    position: Optional[int]
    multiplier: Optional[int]
    is_captain: bool = False
    is_vice_captain: bool = False
    player: UserTeamPlayerInfo


class UserTeamHistoryEntry(BaseModel):
    event: Optional[int]
    points: Optional[int]
    total_points: Optional[int]
    rank: Optional[int]
    overall_rank: Optional[int]
    event_transfers: Optional[int]
    event_transfers_cost: Optional[int]
    bank: Optional[float]
    value: Optional[float]
    points_on_bench: Optional[int]


class UserTeamPastSeason(BaseModel):
    season_name: Optional[str]
    total_points: Optional[int]
    rank: Optional[int]


class UserTeamHistory(BaseModel):
    current: List[UserTeamHistoryEntry]
    past: List[UserTeamPastSeason]


class UserTeamSummary(BaseModel):
    id: Optional[int]
    name: Optional[str]
    player_first_name: Optional[str]
    player_last_name: Optional[str]
    player_region_name: Optional[str]
    summary_overall_points: Optional[int]
    summary_overall_rank: Optional[int]
    summary_event_points: Optional[int]
    summary_event_rank: Optional[int]
    summary_event_transfers: Optional[int]
    summary_event_transfers_cost: Optional[int]
    current_event: Optional[int]
    total_transfers: Optional[int]
    team_value: Optional[float]
    bank: Optional[float]
    favourite_team: Optional[str]
    favourite_team_name: Optional[str] = None
    joined_time: Optional[datetime]


class UserTeamResponse(BaseModel):
    team: UserTeamSummary
    current_event: Optional[int]
    current_event_summary: Optional[UserTeamHistoryEntry]
    picks: List[UserTeamPick]
    bench: List[UserTeamPick]
    chips: List[dict]
    history: UserTeamHistory
    source: str
    fetched_at: datetime