"""
Helper script to load environment variables from .env file
Usage: python load_env.py && python scripts/export_data.py
"""
import os
from pathlib import Path


def load_env_file(env_path=".env"):
    """Load environment variables from .env file"""
    if not Path(env_path).exists():
        print(f"❌ {env_path} file not found!")
        print("Create a .env file with your Supabase credentials:")
        print("SUPABASE_URL=https://your-project-id.supabase.co")
        print("SUPABASE_KEY=your-anon-public-key")
        return False
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
                print(f"✅ Set {key.strip()}")
    
    return True


if __name__ == "__main__":
    load_env_file()