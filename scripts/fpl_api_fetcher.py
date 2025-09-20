"""
FPL Data Fetcher - Direct from Official API
Fetches Fantasy Premier League data directly from the official API
"""
import requests
import pandas as pd
import json
import os
from datetime import datetime
import time

class FPLDataFetcher:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_bootstrap_data(self):
        """Fetch the main bootstrap data containing players, teams, etc."""
        url = f"{self.base_url}/bootstrap-static/"
        print("Fetching bootstrap data...")
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_fixtures(self):
        """Fetch all fixtures for the season"""
        url = f"{self.base_url}/fixtures/"
        print("Fetching fixtures...")
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_player_data(self, player_id):
        """Fetch detailed data for a specific player"""
        url = f"{self.base_url}/element-summary/{player_id}/"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except:
            print(f"Failed to fetch data for player {player_id}")
            return None

    def process_bootstrap_data(self, data):
        """Process the bootstrap data into structured DataFrames"""
        # Extract teams
        teams_df = pd.DataFrame(data['teams'])
        teams_df = teams_df[['id', 'code', 'name', 'short_name', 'strength', 'strength_overall_home',
                           'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
                           'strength_defence_home', 'strength_defence_away']]

        # Extract players
        players_df = pd.DataFrame(data['elements'])
        players_df = players_df[['id', 'first_name', 'second_name', 'web_name', 'code',
                               'team', 'team_code', 'position']]

        # Extract positions
        positions_df = pd.DataFrame(data['element_types'])
        positions_df = positions_df[['id', 'singular_name_short']]

        # Merge position names
        players_df = players_df.merge(positions_df, left_on='position', right_on='id', how='left')
        players_df = players_df.rename(columns={'singular_name_short': 'position_name'})
        players_df = players_df.drop('position', axis=1)

        return teams_df, players_df

    def process_fixtures(self, fixtures_data):
        """Process fixtures data"""
        fixtures_df = pd.DataFrame(fixtures_data)
        if not fixtures_df.empty:
            fixtures_df = fixtures_df[['id', 'event', 'finished', 'kickoff_time', 'team_h', 'team_a',
                                     'team_h_score', 'team_a_score']]
            fixtures_df = fixtures_df.rename(columns={
                'id': 'match_id',
                'event': 'gameweek',
                'team_h': 'home_team',
                'team_a': 'away_team',
                'team_h_score': 'home_score',
                'team_a_score': 'away_score'
            })
        return fixtures_df

    def save_data(self, teams_df, players_df, fixtures_df, output_dir="data/fpl_raw"):
        """Save all data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        teams_df.to_csv(f"{output_dir}/teams.csv", index=False)
        players_df.to_csv(f"{output_dir}/players.csv", index=False)
        fixtures_df.to_csv(f"{output_dir}/fixtures.csv", index=False)

        print(f"Data saved to {output_dir}/")
        print(f"- Teams: {len(teams_df)} records")
        print(f"- Players: {len(players_df)} records")
        print(f"- Fixtures: {len(fixtures_df)} records")

def main():
    print("🏆 FPL Data Fetcher - Getting fresh data from official API")
    print("=" * 60)

    fetcher = FPLDataFetcher()

    try:
        # Fetch data
        bootstrap_data = fetcher.fetch_bootstrap_data()
        fixtures_data = fetcher.fetch_fixtures()

        # Process data
        teams_df, players_df = fetcher.process_bootstrap_data(bootstrap_data)
        fixtures_df = fetcher.process_fixtures(fixtures_data)

        # Save data
        fetcher.save_data(teams_df, players_df, fixtures_df)

        print("\n✅ Data fetch completed successfully!")
        print("You can now use this data or set up Supabase to store it.")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("Make sure you have internet connection and the FPL API is accessible.")

if __name__ == "__main__":
    main()