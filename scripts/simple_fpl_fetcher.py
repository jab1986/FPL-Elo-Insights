"""
Simple FPL Data Fetcher - No Keyboard Dependencies
Fetches Fantasy Premier League data directly from the official API
"""
import requests
import pandas as pd
import os
from datetime import datetime

def fetch_fpl_data():
    """Fetch basic FPL data from the official API"""
    print("🏆 Fetching FPL Data...")

    # Set up session with headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    try:
        # Fetch bootstrap data
        print("📡 Connecting to FPL API...")
        response = session.get('https://fantasy.premierleague.com/api/bootstrap-static/')
        response.raise_for_status()

        data = response.json()
        print(f"✅ Connected! Found {len(data['elements'])} players")

        # Process teams
        teams_data = []
        for team in data['teams']:
            teams_data.append({
                'id': team['id'],
                'code': team['code'],
                'name': team['name'],
                'short_name': team['short_name']
            })

        teams_df = pd.DataFrame(teams_data)
        print(f"📊 Processed {len(teams_df)} teams")

        # Process players
        players_data = []
        for player in data['elements']:
            players_data.append({
                'id': player['id'],
                'first_name': player['first_name'],
                'second_name': player['second_name'],
                'web_name': player['web_name'],
                'team': player['team'],
                'team_code': player['team_code'],
                'element_type': player['element_type']
            })

        players_df = pd.DataFrame(players_data)
        print(f"👥 Processed {len(players_df)} players")

        # Process positions
        positions_data = []
        for pos in data['element_types']:
            positions_data.append({
                'id': pos['id'],
                'name': pos['singular_name_short']
            })

        positions_df = pd.DataFrame(positions_data)

        # Merge position names
        players_df = players_df.merge(positions_df, left_on='element_type', right_on='id', how='left')
        players_df = players_df.rename(columns={'name': 'position'})
        players_df = players_df.drop(['element_type', 'id_y'], axis=1, errors='ignore')

        # Save data
        output_dir = "data/fpl_fresh"
        os.makedirs(output_dir, exist_ok=True)

        teams_df.to_csv(f"{output_dir}/teams.csv", index=False)
        players_df.to_csv(f"{output_dir}/players.csv", index=False)

        print(f"💾 Data saved to {output_dir}/")
        print(f"   - teams.csv: {len(teams_df)} teams")
        print(f"   - players.csv: {len(players_df)} players")

        return teams_df, players_df

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def create_sample_analysis():
    """Create a simple analysis of the fetched data"""
    try:
        teams_df = pd.read_csv("data/fpl_fresh/teams.csv")
        players_df = pd.read_csv("data/fpl_fresh/players.csv")

        print("\n📈 Quick Analysis:")
        print(f"Premier League Teams: {len(teams_df)}")
        print(f"Total Players: {len(players_df)}")

        # Players by position
        position_counts = players_df['position'].value_counts()
        print("\nPlayers by Position:")
        for pos, count in position_counts.items():
            print(f"  {pos}: {count}")

        # Players by team
        team_counts = players_df['team'].value_counts().head(5)
        print("\nTop 5 Teams by Player Count:")
        for team_id, count in team_counts.items():
            team_name = teams_df[teams_df['id'] == team_id]['name'].iloc[0]
            print(f"  {team_name}: {count} players")

    except Exception as e:
        print(f"❌ Analysis error: {e}")

if __name__ == "__main__":
    print("🎯 FPL Data Fetcher - Keyboard-Free Version")
    print("=" * 50)

    teams_df, players_df = fetch_fpl_data()

    if teams_df is not None and players_df is not None:
        create_sample_analysis()
        print("\n✅ Success! Your FPL data is ready.")
        print("You can now use this data for analysis or upload to Supabase.")