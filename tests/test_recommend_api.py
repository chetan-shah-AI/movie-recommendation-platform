from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_recommend_endpoint_schema():
    payload = {
        "user_id": 10011,
        "top_n": 5,
        "genre_filter": None,
        "min_predicted_score": 0,
        "include_explanation": False,
    }

    response = client.post("/recommend", json=payload)

    assert response.status_code in [200, 503]

    body = response.json()

    if response.status_code == 200:
        assert "user_id" in body
        assert "recommendation_type" in body
        assert "recommendation_count" in body
        assert "recommendations" in body
        assert "explanation" in body