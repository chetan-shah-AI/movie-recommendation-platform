from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint_returns_status():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert "status" in body
    assert "recommender_loaded" in body