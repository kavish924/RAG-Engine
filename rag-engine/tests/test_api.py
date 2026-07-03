"""Tests for app/api/routes/*. Cover: /health, /v1/ask, /v1/documents,
/v1/ingest request/response contracts using FastAPI's TestClient."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
