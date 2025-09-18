#!/usr/bin/env python3
"""
Test script for authentication endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_auth_endpoints():
    """Test authentication endpoints"""
    print("Testing Authentication Endpoints...")
    
    # Test signup
    print("\n1. Testing signup endpoint...")
    signup_data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "display_name": "Test User"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Signup successful")
            data = response.json()
            print(f"User: {data['user']['email']}")
            print(f"Token type: {data['token_type']}")
        else:
            print(f"❌ Signup failed: {response.text}")
    except Exception as e:
        print(f"❌ Signup error: {e}")
    
    # Test login
    print("\n2. Testing login endpoint...")
    login_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Login successful")
            data = response.json()
            print(f"User: {data['user']['email']}")
            print(f"Token type: {data['token_type']}")
            return data['access_token']
        else:
            print(f"❌ Login failed: {response.text}")
    except Exception as e:
        print(f"❌ Login error: {e}")
    
    return None

def test_protected_endpoints(token):
    """Test protected endpoints with token"""
    if not token:
        print("\n❌ No token available for protected endpoint tests")
        return
    
    print(f"\n3. Testing protected chat endpoint with token...")
    headers = {"Authorization": f"Bearer {token}"}
    chat_data = {"message": "Hello, this is a test message"}
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Chat endpoint accessible")
            data = response.json()
            print(f"Response: {data['response'][:100]}...")
        else:
            print(f"❌ Chat endpoint failed: {response.text}")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")

def test_public_endpoints():
    """Test public endpoints"""
    print("\n4. Testing public endpoints...")
    
    # Test main page
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Main page status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Main page accessible")
        else:
            print("❌ Main page failed")
    except Exception as e:
        print(f"❌ Main page error: {e}")
    
    # Test auth page
    try:
        response = requests.get(f"{BASE_URL}/auth")
        print(f"Auth page status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Auth page accessible")
        else:
            print("❌ Auth page failed")
    except Exception as e:
        print(f"❌ Auth page error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Authentication Tests")
    print("=" * 50)
    
    # Test public endpoints first
    test_public_endpoints()
    
    # Test authentication (will fail without Firebase setup)
    token = test_auth_endpoints()
    
    # Test protected endpoints
    test_protected_endpoints(token)
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed")
    print("\nNote: Authentication tests will fail without proper Firebase configuration.")
    print("Please set up Firebase service account and environment variables.")
