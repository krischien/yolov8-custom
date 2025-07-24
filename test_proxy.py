#!/usr/bin/env python3
"""
Test webapp proxy functionality
"""

import requests
import time

def test_proxy():
    """Test webapp proxy functionality"""
    
    print("Testing webapp proxy...")
    
    # Test 1: Test the webapp's test-connection endpoint
    try:
        response = requests.get(
            "http://localhost:5001/api/test-connection",
            timeout=5
        )
        print(f"Webapp test-connection: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Webapp test-connection failed: {e}")
    
    # Test 2: Test the API's hello world endpoint directly
    try:
        response = requests.get(
            "http://localhost:5000/hello_world",
            timeout=5
        )
        print(f"API hello_world: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"API hello_world failed: {e}")
    
    # Test 3: Test the webapp's proxy to API hello world
    try:
        response = requests.get(
            "http://localhost:5001/api/hello_world",
            timeout=5
        )
        print(f"Webapp proxy to API: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Webapp proxy failed: {e}")

if __name__ == "__main__":
    print("Proxy Test")
    print("Waiting 2 seconds for services to start...")
    time.sleep(2)
    test_proxy() 