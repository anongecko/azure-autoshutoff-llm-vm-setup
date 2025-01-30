import pytest
from fastapi.testclient import TestClient
from src.api import create_app
import json
import asyncio
from typing import Dict, Any
import time

@pytest.fixture
def client():
    """Create test client"""
    app = create_app()
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests"""
    return {"Authorization": "Bearer sk-test-key"}

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gpu_memory" in data

def test_chat_completions(client, auth_headers):
    """Test chat completions endpoint"""
    request_data = {
        "messages": [
            {"role": "user", "content": "Write a quicksort implementation."}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

def test_streaming_response(client, auth_headers):
    """Test streaming response"""
    request_data = {
        "messages": [
            {"role": "user", "content": "Write a merge sort implementation."}
        ],
        "stream": True
    }
    
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers=auth_headers,
        stream=True
    )
    assert response.status_code == 200
    
    chunks = []
    for line in response.iter_lines():
        if line:
            chunks.append(line)
    assert len(chunks) > 0

def test_concurrent_requests(client, auth_headers):
    """Test handling of concurrent requests"""
    async def make_request() -> Dict[str, Any]:
        request_data = {
            "messages": [
                {"role": "user", "content": "Write a bubble sort implementation."}
            ]
        }
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers=auth_headers
        )
        return response.json()
    
    # Make concurrent requests
    tasks = [make_request() for _ in range(5)]
    responses = asyncio.run(asyncio.gather(*tasks))
    
    assert len(responses) == 5
    for response in responses:
        assert "choices" in response

def test_error_handling(client, auth_headers):
    """Test API error handling"""
    # Test invalid request
    response = client.post(
        "/v1/chat/completions",
        json={"invalid": "request"},
        headers=auth_headers
    )
    assert response.status_code in [400, 422]
    
    # Test missing auth
    response = client.post(
        "/v1/chat/completions",
        json={"messages": []}
    )
    assert response.status_code == 401

def test_rate_limiting(client, auth_headers):
    """Test rate limiting"""
    for _ in range(110):  # Over the rate limit
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
            headers=auth_headers
        )
    
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
        headers=auth_headers
    )
    assert response.status_code == 429

