// FPL Data Types
export interface Player {
  id: number;
  name: string;
  team: string;
  position: string;
  now_cost: number;
  selected_by_percent: number;
  total_points: number;
  minutes: number;
  goals_scored: number;
  assists: number;
  clean_sheets: number;
  goals_conceded: number;
  own_goals: number;
  penalties_saved: number;
  penalties_missed: number;
  yellow_cards: number;
  red_cards: number;
  saves: number;
  bonus: number;
  bps: number;
  influence: number;
  creativity: number;
  threat: number;
  ict_index: number;
  starts: number;
  expected_goals: number;
  expected_assists: number;
  expected_goal_involvements: number;
  expected_goals_conceded: number;
  value_season: number;
  value_form: number;
  points_per_game: number;
  transfers_in: number;
  transfers_out: number;
  transfers_in_event: number;
  transfers_out_event: number;
}

export interface Team {
  id: number;
  name: string;
  short_name: string;
  strength: number;
  played: number;
  win: number;
  draw: number;
  loss: number;
  points: number;
  position: number;
  form: string;
  strength_overall_home: number;
  strength_overall_away: number;
  strength_attack_home: number;
  strength_attack_away: number;
  strength_defence_home: number;
  strength_defence_away: number;
}

export interface Match {
  id: number;
  gameweek: number;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  kickoff_time: string;
  finished: boolean;
  minutes: number;
  provisional_start_time: boolean;
  finished_provisional: boolean;
  started: boolean;
}

export interface PlayerMatchStats {
  id: number;
  match_id: number;
  player_id: number;
  team_id: number;
  position: string;
  minutes: number;
  goals_scored: number;
  assists: number;
  clean_sheets: number;
  goals_conceded: number;
  own_goals: number;
  penalties_saved: number;
  penalties_missed: number;
  yellow_cards: number;
  red_cards: number;
  saves: number;
  bonus: number;
  bps: number;
  influence: number;
  creativity: number;
  threat: number;
  ict_index: number;
  total_points: number;
  in_dreamteam: boolean;
}

export interface GameweekSummary {
  gameweek: number;
  average_entry_score: number;
  highest_score: number;
  deadline_time: string;
  deadline_time_epoch: number;
  finished: boolean;
  data_checked: boolean;
  highest_scoring_entry: number | null;
  most_selected: number | null;
  most_transferred_in: number | null;
  top_element: number | null;
  top_element_info: {
    id: number;
    points: number;
  } | null;
  most_captained: number | null;
  most_vice_captained: number | null;
}

export interface EloRating {
  player_id: number;
  gameweek: number;
  elo_rating: number;
  previous_elo: number;
  change: number;
}

// API Response Types
export interface ApiResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
}

export interface DashboardStats {
  totalPlayers: number;
  totalTeams: number;
  currentGameweek: number;
  averagePoints: number;
  topScorer: Player | null;
  mostValuable: Player | null;
}

// Chart Data Types
export interface ChartDataPoint {
  name: string;
  value: number;
  [key: string]: string | number | null | undefined;
}

export interface TimeSeriesData {
  gameweek: number;
  points: number;
  elo: number;
  date: string;
}

// Filter Types
export interface PlayerFilters {
  position?: string;
  team?: string;
  minPrice?: number;
  maxPrice?: number;
  minPoints?: number;
  maxPoints?: number;
  gameweek?: number;
}

export interface MatchFilters {
  gameweek?: number;
  team?: string;
  finished?: boolean;
}

// User Team Types
export interface UserTeamPlayerInfo {
  id: number | null;
  web_name: string | null;
  first_name: string | null;
  second_name: string | null;
  team: string | null;
  team_short_name: string | null;
  position: string | null;
  now_cost: number | null;
  total_points: number | null;
  selected_by_percent: number | null;
  event_points?: number | null;
  status?: string | null;
}

export interface UserTeamPick {
  element: number | null;
  position: number | null;
  multiplier: number | null;
  is_captain: boolean;
  is_vice_captain: boolean;
  player: UserTeamPlayerInfo;
}

export interface UserTeamHistoryEntry {
  event: number | null;
  points: number | null;
  total_points: number | null;
  rank: number | null;
  overall_rank: number | null;
  event_transfers: number | null;
  event_transfers_cost: number | null;
  bank: number | null;
  value: number | null;
  points_on_bench: number | null;
}

export interface UserTeamPastSeason {
  season_name: string | null;
  total_points: number | null;
  rank: number | null;
}

export interface UserTeamHistory {
  current: UserTeamHistoryEntry[];
  past: UserTeamPastSeason[];
}

export interface UserTeamSummary {
  id: number | null;
  name: string | null;
  player_first_name: string | null;
  player_last_name: string | null;
  player_region_name: string | null;
  summary_overall_points: number | null;
  summary_overall_rank: number | null;
  summary_event_points: number | null;
  summary_event_rank: number | null;
  summary_event_transfers: number | null;
  summary_event_transfers_cost: number | null;
  current_event: number | null;
  total_transfers: number | null;
  team_value: number | null;
  bank: number | null;
  favourite_team: number | string | null;
  favourite_team_name?: string | null;
  joined_time: string | null;
}

export interface UserTeamResponse {
  team: UserTeamSummary;
  current_event: number | null;
  current_event_summary: UserTeamHistoryEntry | null;
  picks: UserTeamPick[];
  bench: UserTeamPick[];
  chips: Array<Record<string, unknown>>;
  history: UserTeamHistory;
  source: 'live' | 'sample';
  fetched_at: string;
}