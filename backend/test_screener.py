
import requests
import json
import sys

BASE_URL = "http://localhost:5001"

def test_screener():
    print("Testing Stock Screener Endpoint...")
    
    # Test 1: No filters (should return all recommendations)
    print("\n1. Testing no filters...")
    try:
        response = requests.get(f"{BASE_URL}/screener")
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {data['count']} stocks.")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Price filter
    print("\n2. Testing price filter (min_price=100)...")
    try:
        response = requests.get(f"{BASE_URL}/screener?min_price=100")
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {data['count']} stocks with price >= 100.")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Score filter
    print("\n3. Testing score filter (min_technical_score=0.1)...")
    try:
        response = requests.get(f"{BASE_URL}/screener?min_technical_score=0.1")
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {data['count']} stocks with technical_score >= 0.1.")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: Combined filters
    print("\n4. Testing combined filters (min_price=100 & min_technical_score=0.1)...")
    try:
        response = requests.get(f"{BASE_URL}/screener?min_price=100&min_technical_score=0.1")
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {data['count']} stocks matching criteria.")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_screener()
