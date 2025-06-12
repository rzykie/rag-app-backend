#!/usr/bin/env python3

import asyncio
import httpx
import requests
from config import settings

async def test_httpx_connection():
    """Test connection using httpx (same as the FastAPI app)"""
    print(f"Testing httpx connection to: {settings.OLLAMA_BASE_URL}")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(settings.OLLAMA_BASE_URL)
            print(f"httpx SUCCESS: Status {response.status_code}")
            return True
    except Exception as e:
        print(f"httpx FAILED: {e}")
        return False

def test_requests_connection():
    """Test connection using requests library"""
    print(f"Testing requests connection to: {settings.OLLAMA_BASE_URL}")
    try:
        response = requests.get(settings.OLLAMA_BASE_URL, timeout=10)
        print(f"requests SUCCESS: Status {response.status_code}")
        return True
    except Exception as e:
        print(f"requests FAILED: {e}")
        return False

async def test_different_urls():
    """Test different URL formats"""
    urls_to_test = [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://0.0.0.0:11434"
    ]
    
    for url in urls_to_test:
        print(f"\nTesting URL: {url}")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                print(f"SUCCESS: Status {response.status_code}")
        except Exception as e:
            print(f"FAILED: {e}")

async def main():
    print("=== Connection Test Results ===")
    print(f"Current OLLAMA_BASE_URL: {settings.OLLAMA_BASE_URL}")
    print()
    
    await test_httpx_connection()
    test_requests_connection()
    
    await test_different_urls()

if __name__ == "__main__":
    asyncio.run(main()) 