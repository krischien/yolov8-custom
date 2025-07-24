#!/usr/bin/env python3
"""
Test script to verify API functionality
"""

import requests
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    print("Testing API endpoints...")
    
    # Test 1: Basic connection
    try:
        response = requests.get(f"{base_url}/hello_world", timeout=5)
        print(f"✓ Hello World: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Hello World failed: {e}")
        return False
    
    # Test 2: Camera preview
    try:
        response = requests.get(f"{base_url}/camera/preview/0", timeout=10)
        print(f"✓ Camera Preview: {response.status_code}")
        if response.status_code != 200:
            print(f"  Response: {response.text[:200]}...")
    except Exception as e:
        print(f"✗ Camera Preview failed: {e}")
    
    # Test 3: Live detection
    try:
        response = requests.get(f"{base_url}/camera/detect_live/0?confidence=0.6", timeout=10)
        print(f"✓ Live Detection: {response.status_code}")
        if response.status_code != 200:
            print(f"  Response: {response.text[:200]}...")
    except Exception as e:
        print(f"✗ Live Detection failed: {e}")
    
    return True

if __name__ == "__main__":
    print("API Test Script")
    print("=" * 50)
    
    # Wait a moment for API to start
    print("Waiting 3 seconds for API to start...")
    time.sleep(3)
    
    success = test_api()
    
    if success:
        print("\n✓ API tests completed")
    else:
        print("\n✗ API tests failed")
        print("\nMake sure:")
        print("1. API is running on port 5000")
        print("2. No firewall blocking the connection")
        print("3. Check API logs for errors") 