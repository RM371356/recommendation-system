"""
Testes de contrato dos endpoints REST.

Estes testes exercitam a API completa via TestClient, mas usando o
fallback natural do serviço (sem artefatos treinados), o que valida
tanto o roteamento quanto o formato do payload.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_expected_contract(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()

    assert set(payload.keys()) >= {
        "status",
        "version",
        "model_loaded",
        "artifacts_loaded",
    }
    assert payload["status"] == "healthy"
    assert isinstance(payload["model_loaded"], bool)
    assert isinstance(payload["artifacts_loaded"], bool)


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------


def test_recommend_endpoint_returns_expected_contract(api_client: TestClient) -> None:
    """
    Sem artefatos, o serviço deve cair no fallback de popularidade
    e ainda assim devolver um payload no formato esperado.
    """

    response = api_client.post(
        "/api/v1/recommend",
        json={"user_id": 1, "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["user_id"] == 1
    assert payload["total"] == len(payload["recommendations"])
    assert len(payload["recommendations"]) == 5

    for item in payload["recommendations"]:
        assert {"rank", "movie_id", "title", "genres", "predicted_score"}.issubset(item)
        assert isinstance(item["rank"], int)
        assert isinstance(item["movie_id"], int)
        assert isinstance(item["predicted_score"], float)


def test_recommend_default_top_k_returns_ten_items(api_client: TestClient) -> None:
    """Quando top_k é omitido, o default (10) deve ser aplicado."""

    response = api_client.post("/api/v1/recommend", json={"user_id": 42})

    assert response.status_code == 200
    assert response.json()["total"] == 10


def test_recommend_ranks_are_monotonically_increasing(api_client: TestClient) -> None:
    """Os ranks devem começar em 1 e crescer sem gaps."""

    response = api_client.post(
        "/api/v1/recommend", json={"user_id": 1, "top_k": 7}
    )
    ranks = [item["rank"] for item in response.json()["recommendations"]]

    assert ranks == list(range(1, 8))
