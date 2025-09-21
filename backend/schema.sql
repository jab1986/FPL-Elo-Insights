-- FPL Insights Database Schema for Supabase
-- Run this SQL in Supabase SQL editor to create all required tables

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS gameweek_summaries CASCADE;
DROP TABLE IF EXISTS playermatchstats CASCADE;
DROP TABLE IF EXISTS matches CASCADE;
DROP TABLE IF EXISTS players CASCADE;
DROP TABLE IF EXISTS teams CASCADE;

-- Teams table
CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    short_name VARCHAR(10) NOT NULL,
    strength INTEGER NOT NULL DEFAULT 0,
    played INTEGER NOT NULL DEFAULT 0,
    win INTEGER NOT NULL DEFAULT 0,
    draw INTEGER NOT NULL DEFAULT 0,
    loss INTEGER NOT NULL DEFAULT 0,
    points INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    form VARCHAR(50),
    strength_overall_home INTEGER NOT NULL DEFAULT 0,
    strength_overall_away INTEGER NOT NULL DEFAULT 0,
    strength_attack_home INTEGER NOT NULL DEFAULT 0,
    strength_attack_away INTEGER NOT NULL DEFAULT 0,
    strength_defence_home INTEGER NOT NULL DEFAULT 0,
    strength_defence_away INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Players table
CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    team VARCHAR(100) NOT NULL,
    position VARCHAR(20) NOT NULL,
    now_cost INTEGER NOT NULL DEFAULT 0,
    selected_by_percent DECIMAL(5,2) NOT NULL DEFAULT 0.0,
    total_points INTEGER NOT NULL DEFAULT 0,
    minutes INTEGER NOT NULL DEFAULT 0,
    goals_scored INTEGER NOT NULL DEFAULT 0,
    assists INTEGER NOT NULL DEFAULT 0,
    clean_sheets INTEGER NOT NULL DEFAULT 0,
    goals_conceded INTEGER NOT NULL DEFAULT 0,
    own_goals INTEGER NOT NULL DEFAULT 0,
    penalties_saved INTEGER NOT NULL DEFAULT 0,
    penalties_missed INTEGER NOT NULL DEFAULT 0,
    yellow_cards INTEGER NOT NULL DEFAULT 0,
    red_cards INTEGER NOT NULL DEFAULT 0,
    saves INTEGER NOT NULL DEFAULT 0,
    bonus INTEGER NOT NULL DEFAULT 0,
    bps INTEGER NOT NULL DEFAULT 0,
    influence DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    creativity DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    threat DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    ict_index DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    starts INTEGER NOT NULL DEFAULT 0,
    expected_goals DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    expected_assists DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    expected_goal_involvements DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    expected_goals_conceded DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    value_season DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    value_form DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    points_per_game DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    transfers_in INTEGER NOT NULL DEFAULT 0,
    transfers_out INTEGER NOT NULL DEFAULT 0,
    transfers_in_event INTEGER NOT NULL DEFAULT 0,
    transfers_out_event INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Matches table
CREATE TABLE matches (
    id INTEGER PRIMARY KEY,
    gameweek INTEGER NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    kickoff_time TIMESTAMP WITH TIME ZONE NOT NULL,
    finished BOOLEAN NOT NULL DEFAULT FALSE,
    minutes INTEGER NOT NULL DEFAULT 0,
    provisional_start_time BOOLEAN NOT NULL DEFAULT FALSE,
    finished_provisional BOOLEAN NOT NULL DEFAULT FALSE,
    started BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Player Match Stats table
CREATE TABLE playermatchstats (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    position VARCHAR(20) NOT NULL,
    minutes INTEGER NOT NULL DEFAULT 0,
    goals_scored INTEGER NOT NULL DEFAULT 0,
    assists INTEGER NOT NULL DEFAULT 0,
    clean_sheets INTEGER NOT NULL DEFAULT 0,
    goals_conceded INTEGER NOT NULL DEFAULT 0,
    own_goals INTEGER NOT NULL DEFAULT 0,
    penalties_saved INTEGER NOT NULL DEFAULT 0,
    penalties_missed INTEGER NOT NULL DEFAULT 0,
    yellow_cards INTEGER NOT NULL DEFAULT 0,
    red_cards INTEGER NOT NULL DEFAULT 0,
    saves INTEGER NOT NULL DEFAULT 0,
    bonus INTEGER NOT NULL DEFAULT 0,
    bps INTEGER NOT NULL DEFAULT 0,
    influence DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    creativity DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    threat DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    ict_index DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    total_points INTEGER NOT NULL DEFAULT 0,
    in_dreamteam BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(match_id, player_id)
);

-- Gameweek Summaries table
CREATE TABLE gameweek_summaries (
    gameweek INTEGER PRIMARY KEY,
    average_entry_score DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    highest_score INTEGER NOT NULL DEFAULT 0,
    deadline_time TIMESTAMP WITH TIME ZONE NOT NULL,
    deadline_time_epoch INTEGER NOT NULL,
    finished BOOLEAN NOT NULL DEFAULT FALSE,
    data_checked BOOLEAN NOT NULL DEFAULT FALSE,
    highest_scoring_entry INTEGER,
    most_selected INTEGER,
    most_transferred_in INTEGER,
    top_element INTEGER,
    top_element_info JSONB,
    most_captained INTEGER,
    most_vice_captained INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_players_team ON players(team);
CREATE INDEX idx_players_position ON players(position);
CREATE INDEX idx_players_total_points ON players(total_points DESC);
CREATE INDEX idx_matches_gameweek ON matches(gameweek);
CREATE INDEX idx_matches_finished ON matches(finished);
CREATE INDEX idx_playermatchstats_match_id ON playermatchstats(match_id);
CREATE INDEX idx_playermatchstats_player_id ON playermatchstats(player_id);
CREATE INDEX idx_playermatchstats_team_id ON playermatchstats(team_id);

-- Enable Row Level Security (RLS) for all tables
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE players ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE playermatchstats ENABLE ROW LEVEL SECURITY;
ALTER TABLE gameweek_summaries ENABLE ROW LEVEL SECURITY;

-- Create policies to allow read access to all authenticated users
CREATE POLICY "Allow read access to teams" ON teams FOR SELECT USING (true);
CREATE POLICY "Allow read access to players" ON players FOR SELECT USING (true);
CREATE POLICY "Allow read access to matches" ON matches FOR SELECT USING (true);
CREATE POLICY "Allow read access to playermatchstats" ON playermatchstats FOR SELECT USING (true);
CREATE POLICY "Allow read access to gameweek_summaries" ON gameweek_summaries FOR SELECT USING (true);

-- Create policies to allow insert/update for service role
CREATE POLICY "Allow service role to manage teams" ON teams FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Allow service role to manage players" ON players FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Allow service role to manage matches" ON matches FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Allow service role to manage playermatchstats" ON playermatchstats FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Allow service role to manage gameweek_summaries" ON gameweek_summaries FOR ALL USING (auth.role() = 'service_role');