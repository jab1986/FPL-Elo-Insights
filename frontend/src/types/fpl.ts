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