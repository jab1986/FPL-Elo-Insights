#!/usr/bin/env python3

import os
from dotenv import load_dotenv

# Test loading environment variables from parent directory
print("Testing environment variable loading...")

# Load from parent directory (where .env should be)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
print(f"Looking for .env at: {os.path.abspath(env_path)}")
print(f".env file exists: {os.path.exists(env_path)}")

# Load the environment variables
load_dotenv(env_path)

# Check if variables are loaded
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

print(f"SUPABASE_URL: {supabase_url}")
print(f"SUPABASE_KEY found: {bool(supabase_key)}")
print(f"SUPABASE_KEY length: {len(supabase_key) if supabase_key else 0}")

if supabase_url and supabase_key:
    print("✅ Supabase credentials loaded successfully!")
    
    # Test if we can create a Supabase client
    try:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)
        print("✅ Supabase client created successfully!")
        
        # Test a simple connection
        try:
            response = client.table('players').select('id').limit(1).execute()
            print(f"✅ Database connection test successful! Found {len(response.data)} record(s)")
        except Exception as e:
            print(f"❌ Database connection test failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to create Supabase client: {e}")
else:
    print("❌ Supabase credentials not found")
