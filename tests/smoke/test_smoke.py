"""
Smoke tests

Objetivo: garantir que o "esqueleto" do sistema não está quebrado.
Se qualquer destes falhar, é sinal de que o build está impróprio para
qualquer teste mais aprofundado — por isso rodam primeiro (marker `smoke`).
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Imports básicos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "src.api.main",
        "src.api.routes",
        "src.api.schemas",
        "src.api.exception_handlers",
        "src.api.dependencies",
        "src.services.recommendation_service",
        "src.inference.cache",
        "src.inference.predictor",
        "src.inference.model_loader",
        "src.inference.encoder_loader",
        "src.inference.movie_repository",
        "src.models.mlp_model",
        "src.models.model_factory",
        "src.models.baseline_model",
        "src.data.dataset",
        "src.data.preprocess",
        "src.features.negative_sampling",
        "src.evaluation.metrics",
        "src.training.early_stopping",
        "src.utils.logger",
        "src.utils.seed",
        "src.config.settings",
    ],
)
def test_module_imports_without_errors(module_path: str) -> None:
    """Qualquer módulo do pacote src deve ser importável sem side effects fatais."""

    module = importlib.import_module(module_path)
    assert module is not None


# ---------------------------------------------------------------------------
# App FastAPI sobe
# ---------------------------------------------------------------------------


def test_app_boots_and_serves_health() -> None:
    """
    Boot completo da API + primeira resposta do /health.
    Este é o smoke test mais valioso: valida lifespan, routers e schemas.
    """

    from src.api.main import app

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "version" in payload
    assert "model_loaded" in payload
    assert "artifacts_loaded" in payload


def test_openapi_schema_is_available(api_client: TestClient) -> None:
    """Documentação OpenAPI/Swagger deve estar servida."""

    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Recommendation API"
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/recommend" in schema["paths"]
