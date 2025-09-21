# FPL Insights Database Setup

This guide will help you set up the Supabase database with live FPL data.

## Prerequisites

1. Supabase account and project
2. Environment variables configured (`.env` file with `SUPABASE_URL` and `SUPABASE_KEY`)
3. Python dependencies installed (`pip install -r requirements.txt`)

## Step 1: Create Database Schema

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Copy and paste the contents of `schema.sql` into the SQL editor
4. Click **Run** to execute the schema creation

This will create the following tables:
- `teams` - Premier League teams and their statistics
- `players` - Player information and performance data
- `matches` - Match fixtures and results
- `playermatchstats` - Player performance in specific matches
- `gameweek_summaries` - Gameweek summary statistics

## Step 2: Populate Database with Live Data

Run the population script to fetch live data from the FPL API:

```bash
# Make sure you're in the backend directory and virtual environment is activated
cd backend
source venv/bin/activate

# Run the population script
python populate_database.py
```

This script will:
- Clear any existing data
- Fetch live data from the official FPL API (`https://fantasy.premierleague.com/api/`)
- Populate all tables with current season data
- Handle team/player mappings automatically

## Step 3: Verify Setup

Test that the CLI can now access live data:

```bash
# Test dashboard stats
python cli.py dashboard stats

# Test player data
python cli.py players top -l 5

# Test team data
python cli.py teams --help
```

You should now see live FPL data instead of mock data.

## Data Refresh

To refresh the database with the latest FPL data, simply run the population script again:

```bash
python populate_database.py
```

The script will clear existing data and repopulate with fresh data from the API.

## Troubleshooting

### "Could not find table" errors
- Make sure you've run the `schema.sql` in Supabase SQL Editor
- Check that your Supabase credentials are correct in `.env`

### "SUPABASE_URL and SUPABASE_KEY not found" warnings
- Ensure `.env` file is in the correct directory (`backend/` folder)
- Check that environment variables are properly formatted

### API errors during population
- Check your internet connection
- The FPL API might be temporarily unavailable (try again later)
- Ensure the FPL API endpoints are still valid

## Database Structure

### Core Tables
- **teams**: Team data with strength ratings and form
- **players**: Comprehensive player statistics and performance metrics
- **matches**: Fixture list with results and match details
- **gameweek_summaries**: Gameweek statistics and deadline information

### Features
- Row Level Security (RLS) enabled for data protection
- Optimized indexes for common queries
- Foreign key relationships for data integrity
- JSON support for complex data (top_element_info in gameweeks)

## API Rate Limits

The FPL API is publicly available but should be used responsibly:
- Don't run the population script too frequently (once per day is sufficient)
- The script includes reasonable delays between requests
- Official FPL data updates typically happen after matches and during maintenance windows