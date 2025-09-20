import urllib.request
import json


def test_api_endpoint(url):
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            print(f"✅ {url}: {data}")
            return True
    except Exception as e:
        print(f"❌ {url}: {e}")
        return False


# Test endpoints
base_url = "http://localhost:8001"

print("Testing FPL Insights API endpoints...")
print("=" * 50)

# Test root endpoint
test_api_endpoint(f"{base_url}/")

# Test health endpoint
test_api_endpoint(f"{base_url}/health")

# Test API endpoints
test_api_endpoint(f"{base_url}/api/dashboard/stats")

print("=" * 50)
print("API testing complete!")
