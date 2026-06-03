"""API integration tests using FastAPI TestClient."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_status_endpoint(test_client: TestClient) -> None:
    """GET /status should return HTTP 200."""
    response = test_client.get("/status")
    assert response.status_code == 200


def test_search_endpoint(test_client: TestClient) -> None:
    """GET /api/search should return HTTP 200 for a text query."""
    response = test_client.get("/api/search", params={"q": "test", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert "time_ms" in body


def test_register_user(test_client: TestClient) -> None:
    """POST /auth/register should create a user and return 200."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    response = test_client.post(
        "/auth/register",
        json={"username": username, "password": "secretpass"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data.get("message")


def test_login_user(test_client: TestClient) -> None:
    """POST /auth/login should return an access token."""
    username = f"login_{uuid.uuid4().hex[:8]}"
    test_client.post(
        "/auth/register",
        json={"username": username, "password": "testpass123"},
    )
    response = test_client.post(
        "/auth/login",
        json={"username": username, "password": "testpass123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("access_token")
    assert data.get("token_type") == "bearer"


def test_protected_route_without_token(test_client: TestClient) -> None:
    """POST /doc/insert without a token should return 401."""
    response = test_client.post(
        "/doc/insert",
        json={"title": "Test", "text": "Some document text here."},
    )
    assert response.status_code == 401


def test_protected_route_with_token(test_client: TestClient) -> None:
    """POST /doc/insert with a valid token should return 200."""
    username = f"doc_{uuid.uuid4().hex[:8]}"
    test_client.post(
        "/auth/register",
        json={"username": username, "password": "docpass123"},
    )
    login = test_client.post(
        "/auth/login",
        json={"username": username, "password": "docpass123"},
    )
    token = login.json()["access_token"]

    response = test_client.post(
        "/doc/insert",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Protected Doc",
            "text": "This is a long enough document body for chunking tests.",
        },
    )
    assert response.status_code == 200
    assert response.json().get("chunks_inserted", 0) >= 1
