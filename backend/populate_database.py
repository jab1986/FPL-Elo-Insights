#!/usr/bin/env python3
"""
Script to populate Supabase database with live FPL data.
This script fetches data from the official FPL API and inserts it into the database.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

import httpx
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

class FPLDataPopulator:
    """Handles fetching FPL data and populating the database."""

    FPL_BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

        self.supabase: Client = create_client(supabase_url, supabase_key)

    async def fetch_bootstrap_data(self) -> Dict[str, Any]:
        """Fetch bootstrap-static data from FPL API."""
        print("Fetching bootstrap data from FPL API...")

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.FPL_BASE_URL}/bootstrap-static/")
            response.raise_for_status()
            return response.json()

    def clear_existing_data(self):
        """Clear existing data from all tables."""
        print("Clearing existing data...")

        tables = ['playermatchstats', 'matches', 'players', 'teams', 'gameweek_summaries']
        for table in tables:
            try:
                result = self.supabase.table(table).delete().neq('id', 0).execute()
                print(f"Cleared {table} table")
            except Exception as e:
                print(f"Warning: Could not clear {table}: {e}")

    def populate_teams(self, teams_data: List[Dict[str, Any]]):
        """Populate teams table with FPL team data."""
        print("Populating teams table...")

        teams_to_insert = []
        for team in teams_data:
            team_record = {
                'id': team['id'],
                'name': team['name'],
                'short_name': team['short_name'],
                'strength': team.get('strength', 0),
                'played': team.get('played', 0),
                'win': team.get('win', 0),
                'draw': team.get('draw', 0),
                'loss': team.get('loss', 0),
                'points': team.get('points', 0),
                'position': team.get('position', 0),
                'form': team.get('form', ''),
                'strength_overall_home': team.get('strength_overall_home', 0),
                'strength_overall_away': team.get('strength_overall_away', 0),
                'strength_attack_home': team.get('strength_attack_home', 0),
                'strength_attack_away': team.get('strength_attack_away', 0),
                'strength_defence_home': team.get('strength_defence_home', 0),
                'strength_defence_away': team.get('strength_defence_away', 0)
            }
            teams_to_insert.append(team_record)

        # Insert in batches
        batch_size = 20
        for i in range(0, len(teams_to_insert), batch_size):
            batch = teams_to_insert[i:i + batch_size]
            result = self.supabase.table('teams').insert(batch).execute()
            print(f"Inserted {len(batch)} teams")

    def populate_players(self, players_data: List[Dict[str, Any]], teams_data: List[Dict[str, Any]]):
        """Populate players table with FPL player data."""
        print("Populating players table...")

        # Create team lookup
        team_lookup = {team['id']: team['name'] for team in teams_data}

        players_to_insert = []
        for player in players_data:
            team_name = team_lookup.get(player['team'], 'Unknown')

            player_record = {
                'id': player['id'],
                'name': f"{player['first_name']} {player['second_name']}",
                'team': team_name,
                'position': self._get_position_name(player['element_type']),
                'now_cost': player.get('now_cost', 0),
                'selected_by_percent': float(player.get('selected_by_percent', 0)),
                'total_points': player.get('total_points', 0),
                'minutes': player.get('minutes', 0),
                'goals_scored': player.get('goals_scored', 0),
                'assists': player.get('assists', 0),
                'clean_sheets': player.get('clean_sheets', 0),
                'goals_conceded': player.get('goals_conceded', 0),
                'own_goals': player.get('own_goals', 0),
                'penalties_saved': player.get('penalties_saved', 0),
                'penalties_missed': player.get('penalties_missed', 0),
                'yellow_cards': player.get('yellow_cards', 0),
                'red_cards': player.get('red_cards', 0),
                'saves': player.get('saves', 0),
                'bonus': player.get('bonus', 0),
                'bps': player.get('bps', 0),
                'influence': float(player.get('influence', 0)),
                'creativity': float(player.get('creativity', 0)),
                'threat': float(player.get('threat', 0)),
                'ict_index': float(player.get('ict_index', 0)),
                'starts': player.get('starts', 0),
                'expected_goals': float(player.get('expected_goals', 0)),
                'expected_assists': float(player.get('expected_assists', 0)),
                'expected_goal_involvements': float(player.get('expected_goal_involvements', 0)),
                'expected_goals_conceded': float(player.get('expected_goals_conceded', 0)),
                'value_season': float(player.get('value_season', 0)),
                'value_form': float(player.get('value_form', 0)),
                'points_per_game': float(player.get('points_per_game', 0)),
                'transfers_in': player.get('transfers_in', 0),
                'transfers_out': player.get('transfers_out', 0),
                'transfers_in_event': player.get('transfers_in_event', 0),
                'transfers_out_event': player.get('transfers_out_event', 0)
            }
            players_to_insert.append(player_record)

        # Insert in batches
        batch_size = 50
        for i in range(0, len(players_to_insert), batch_size):
            batch = players_to_insert[i:i + batch_size]
            result = self.supabase.table('players').insert(batch).execute()
            print(f"Inserted {len(batch)} players")

    def populate_gameweeks(self, gameweeks_data: List[Dict[str, Any]]):
        """Populate gameweek_summaries table with FPL gameweek data."""
        print("Populating gameweek summaries table...")

        gameweeks_to_insert = []
        for gw in gameweeks_data:
            gameweek_record = {
                'gameweek': gw['id'],
                'average_entry_score': float(gw.get('average_entry_score', 0)),
                'highest_score': gw.get('highest_score') or 0,
                'deadline_time': gw['deadline_time'],
                'deadline_time_epoch': gw.get('deadline_time_epoch') or 0,
                'finished': gw.get('finished', False),
                'data_checked': gw.get('data_checked', False),
                'highest_scoring_entry': gw.get('highest_scoring_entry'),
                'most_selected': gw.get('most_selected'),
                'most_transferred_in': gw.get('most_transferred_in'),
                'top_element': gw.get('top_element'),
                'top_element_info': gw.get('top_element_info'),
                'most_captained': gw.get('most_captained'),
                'most_vice_captained': gw.get('most_vice_captained')
            }
            gameweeks_to_insert.append(gameweek_record)

        # Insert in batches
        batch_size = 20
        for i in range(0, len(gameweeks_to_insert), batch_size):
            batch = gameweeks_to_insert[i:i + batch_size]
            result = self.supabase.table('gameweek_summaries').insert(batch).execute()
            print(f"Inserted {len(batch)} gameweek summaries")

    async def fetch_and_populate_fixtures(self):
        """Fetch and populate match fixtures."""
        print("Fetching fixtures from FPL API...")

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.FPL_BASE_URL}/fixtures/")
            response.raise_for_status()
            fixtures_data = response.json()

        # Get team lookup for converting team IDs to names
        teams_result = self.supabase.table('teams').select('*').execute()
        team_lookup = {team['id']: team['name'] for team in teams_result.data}

        print("Populating matches table...")
        matches_to_insert = []
        for fixture in fixtures_data:
            match_record = {
                'id': fixture['id'],
                'gameweek': fixture.get('event', 0),
                'home_team': team_lookup.get(fixture['team_h'], 'Unknown'),
                'away_team': team_lookup.get(fixture['team_a'], 'Unknown'),
                'home_score': fixture.get('team_h_score'),
                'away_score': fixture.get('team_a_score'),
                'kickoff_time': fixture['kickoff_time'],
                'finished': fixture.get('finished', False),
                'minutes': fixture.get('minutes', 0),
                'provisional_start_time': fixture.get('provisional_start_time', False),
                'finished_provisional': fixture.get('finished_provisional', False),
                'started': fixture.get('started', False)
            }
            matches_to_insert.append(match_record)

        # Insert in batches
        batch_size = 50
        for i in range(0, len(matches_to_insert), batch_size):
            batch = matches_to_insert[i:i + batch_size]
            result = self.supabase.table('matches').insert(batch).execute()
            print(f"Inserted {len(batch)} matches")

    def _get_position_name(self, element_type: int) -> str:
        """Convert FPL element_type to position name."""
        position_map = {
            1: "GKP",
            2: "DEF",
            3: "MID",
            4: "FWD"
        }
        return position_map.get(element_type, "Unknown")

    async def run_full_population(self):
        """Run the complete data population process."""
        try:
            print("Starting FPL data population...")

            # Clear existing data
            self.clear_existing_data()

            # Fetch bootstrap data
            bootstrap_data = await self.fetch_bootstrap_data()

            # Populate tables
            self.populate_teams(bootstrap_data['teams'])
            self.populate_players(bootstrap_data['elements'], bootstrap_data['teams'])
            self.populate_gameweeks(bootstrap_data['events'])

            # Fetch and populate fixtures
            await self.fetch_and_populate_fixtures()

            print("✅ Database population completed successfully!")
            print("\nData populated:")
            print(f"- {len(bootstrap_data['teams'])} teams")
            print(f"- {len(bootstrap_data['elements'])} players")
            print(f"- {len(bootstrap_data['events'])} gameweeks")
            print("- Match fixtures from current season")

        except Exception as e:
            print(f"❌ Error during population: {e}")
            sys.exit(1)

async def main():
    """Main function to run the population script."""
    populator = FPLDataPopulator()
    await populator.run_full_population()

if __name__ == "__main__":
    asyncio.run(main())