"""
Testes de resiliência / tratamento de erros.

Garantem que o sistema:
- Não trava se artefatos estiverem ausentes
- Retorna fallback de popularidade em vez de crashar
- Faz raise de erros específicos e claros nos loaders individuais
- Exception handlers convertem exceções em respostas JSON
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.exception_handlers import (
    RecommendationException,
    register_exception_handlers,
)
from src.inference.encoder_loader import EncoderLoader
from src.inference.model_loader import ModelLoader
from src.inference.movie_repository import MovieRepository
from src.services.recommendation_service import RecommendationService

pytestmark = pytest.mark.resilience


# ---------------------------------------------------------------------------
# Loaders com arquivos ausentes
# ---------------------------------------------------------------------------


def test_model_loader_raises_when_config_missing(tmp_path: Path) -> None:
    loader = ModelLoader(artifacts_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Config not found"):
        loader.load()


def test_encoder_loader_raises_when_encoder_missing(tmp_path: Path) -> None:
    loader = EncoderLoader(artifacts_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="User encoder not found"):
        loader.load()


def test_movie_repository_raises_when_csv_missing(tmp_path: Path) -> None:
    repo = MovieRepository(artifacts_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Movies metadata not found"):
        repo.load()


def test_movie_repository_raises_for_invalid_schema(tmp_path: Path) -> None:
    """CSV sem colunas obrigatórias deve ser detectado."""

    import pandas as pd

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(metadata_dir / "movies.csv", index=False)

    repo = MovieRepository(artifacts_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Missing columns"):
        repo.load()


# ---------------------------------------------------------------------------
# Serviço em modo fallback
# ---------------------------------------------------------------------------


def test_service_returns_popularity_fallback_without_model() -> None:
    """
    Sem artefatos e sem features.csv, o serviço deve retornar o fallback
    absoluto (lista hardcoded de "Popular Movie N").
    """

    service = RecommendationService()
    response = service.recommend(user_id=1, top_k=5)

    assert response.total == 5
    assert len(response.recommendations) == 5
    titles = [item.title for item in response.recommendations]
    assert all(t.startswith("Popular Movie ") for t in titles)


def test_service_fallback_respects_top_k() -> None:
    service = RecommendationService()

    response_1 = service.recommend(user_id=1, top_k=1)
    response_10 = service.recommend(user_id=1, top_k=10)

    assert response_1.total == 1
    assert response_10.total == 10


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _build_exception_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom-domain")
    def _boom_domain():
        raise RecommendationException("domain error", status_code=418)

    @app.get("/boom-generic")
    def _boom_generic():
        raise RuntimeError("unexpected")

    return app


def test_recommendation_exception_returns_custom_status() -> None:
    client = TestClient(_build_exception_test_app())
    response = client.get("/boom-domain")

    assert response.status_code == 418
    assert response.json() == {"error": "domain error"}


def test_generic_exception_returns_500_json() -> None:
    """Qualquer exceção não-tratada deve virar 500 estruturado em JSON."""

    app = _build_exception_test_app()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom-generic")

    assert response.status_code == 500
    body = response.json()
    assert "error" in body
