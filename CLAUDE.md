# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPL-Elo-Insights is a Fantasy Premier League data analysis platform combining official FPL API data with detailed match statistics and historical team Elo ratings. The architecture follows a full-stack pattern with FastAPI backend, React frontend, and comprehensive CLI tooling.

## Core Development Commands

### Backend Development
```bash
cd backend
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8001
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev      # Development server on :5173
npm run build    # Production build
npm run preview  # Preview production build
```

### CLI Usage
```bash
cd backend
source venv/bin/activate
python cli.py --help                    # View all commands
python cli.py dashboard stats           # Dashboard overview
python cli.py players top -l 10         # Top 10 players
python cli.py user-team show 123456     # Analyze FPL team
```

### Database Setup
```bash
# 1. Run schema.sql in Supabase SQL Editor
# 2. Populate with live data:
cd backend
python populate_database.py
```

### Testing
```bash
# Backend API testing
cd backend
python test_api.py

# Frontend testing
cd frontend
npm run test
```

## Architecture Overview

### Backend (FastAPI)
- **Entry Point**: `backend/main.py`
- **Route Structure**: `backend/app/routes/` - organized by resource (players, teams, matches, dashboard)
- **Service Layer**: `backend/app/services/` - business logic and external API integration
- **Models**: `backend/app/models/__init__.py` - Pydantic models for type safety
- **CLI Tool**: `backend/cli.py` - comprehensive command-line interface

### Frontend (React + TypeScript)
- **Entry Point**: `frontend/src/main.tsx`
- **Pages**: `frontend/src/pages/` - route-level components
- **Components**: `frontend/src/components/` - reusable UI components
- **API Layer**: `frontend/src/services/api.ts` - backend integration
- **Types**: `frontend/src/types/` - TypeScript definitions

### Data Architecture
- **Live Sources**: FPL API (`https://fantasy.premierleague.com/api/`)
- **Database**: Supabase (PostgreSQL) with tables: players, teams, matches, playermatchstats, gameweek_summaries
- **Fallback**: Mock data service when live services unavailable
- **CSV Exports**: `data/` directory with historical datasets organized by season/gameweek

## Key Architectural Patterns

### Data Flow Strategy
1. **Primary**: Live FPL API → Supabase → Backend API → Frontend
2. **Fallback**: Mock data when services unavailable
3. **CLI**: Direct database access with same fallback logic

### Service Layer Design
- `data_service.py` - Supabase database operations with graceful fallback to mock data
- `fpl_service.py` - Official FPL API integration with caching and rate limiting
- `mock_data.py` - Development/fallback data matching production schemas

### Error Handling Philosophy
Services degrade gracefully with clear user feedback when:
- Database connection fails → falls back to mock data
- FPL API unavailable → uses cached/sample data
- Missing environment variables → development mode with warnings

## Database Schema

### Core Tables
- **players** - FPL player data with performance metrics, pricing, and ownership
- **teams** - Premier League teams with Elo ratings and strength metrics
- **matches** - Fixtures, results, and detailed match statistics
- **playermatchstats** - Per-match player performance data
- **gameweek_summaries** - Gameweek metadata, deadlines, and top performers

### Key Relationships
- Players belong to teams (many-to-one)
- Matches involve two teams (home/away)
- PlayerMatchStats links players to specific match performances
- All data organized by gameweeks for temporal analysis

## Environment Configuration

### Required Environment Variables (.env)
```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key

# Optional: Season for merge operations
MERGE_SEASON=2025-2026
```

### File Locations
- Backend: `backend/.env`
- Root: `.env` (copied from root for CLI access)

## Development Workflow

### Starting Development
1. **Backend First**: Set up virtual environment, install dependencies, start server
2. **Database**: Run schema.sql in Supabase, then populate_database.py
3. **Frontend**: Install npm dependencies, start dev server
4. **Verification**: Use CLI tool to test data access

### Working with Data
- **CLI Tool**: Best way to explore and understand data structures
- **Mock Data**: Develop against `mock_data.py` when services unavailable
- **Live Data**: Populate database with `populate_database.py` for real data
- **CSV Exports**: Historical data available in `data/` directory

### Code Organization Principles
- **API Routes**: One file per resource type (players, teams, matches)
- **Type Safety**: Pydantic models shared between CLI and API
- **Separation**: Clear boundaries between data access, business logic, and presentation
- **Fallbacks**: Every service has offline/mock capability

## Special Considerations

### FPL API Integration
- Official API has no authentication but should be used responsibly
- Rate limiting implemented in `fpl_service.py`
- Data updates typically happen after matches and during maintenance windows
- Bootstrap endpoint provides comprehensive player/team/fixture data

### Performance Considerations
- Database queries optimized with indexes on common filters (team, position, gameweek)
- Frontend uses TanStack Query for caching and background updates
- CLI tool supports parallel operations for bulk data operations

### Data Consistency
- Point-in-time snapshots preserve historical accuracy
- Gameweek-based organization allows temporal analysis
- Delta calculations solve cumulative statistics problems
- Foreign key constraints maintain referential integrity

## Common Development Tasks

### Adding New API Endpoints
1. Create route handler in appropriate `backend/app/routes/` file
2. Add Pydantic models to `backend/app/models/__init__.py`
3. Implement service logic in `backend/app/services/data_service.py`
4. Add CLI command in `backend/cli.py` if needed
5. Update frontend API service and types

### Database Schema Changes
1. Update `schema.sql` with new structure
2. Modify Pydantic models to match
3. Update `populate_database.py` if new data sources needed
4. Test with `python cli.py` commands

### Frontend Component Development
1. Create component in appropriate directory (`pages/` vs `components/`)
2. Add TypeScript types in `types/` directory
3. Integrate with API using existing patterns in `services/api.ts`
4. Use TanStack Query for data fetching and caching