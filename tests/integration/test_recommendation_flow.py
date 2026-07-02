"""
Testes de integração do fluxo de recomendação.

Cobrem o pipeline: ModelLoader + EncoderLoader + MovieRepository +
Predictor + RecommendationService — todos em conjunto, usando artefatos
fake fornecidos pela fixture `fake_artifacts_dir`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.encoder_loader import EncoderLoader
from src.inference.model_loader import ModelLoader
from src.inference.movie_repository import MovieRepository
from src.inference.predictor import Predictor
from src.services.recommendation_service import RecommendationService

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Loaders individualmente
# ---------------------------------------------------------------------------


def test_model_loader_can_load_fake_artifacts(fake_artifacts_dir: Path) -> None:
    loader = ModelLoader(artifacts_dir=str(fake_artifacts_dir))
    model = loader.load()

    assert loader.is_loaded is True
    assert model is not None


def test_encoder_loader_can_encode_and_decode(fake_artifacts_dir: Path) -> None:
    loader = EncoderLoader(artifacts_dir=str(fake_artifacts_dir))
    loader.load()

    encoded = loader.encode_user(10)
    decoded = loader.decode_user(encoded)

    assert decoded == 10


def test_movie_repository_lookup_works(fake_artifacts_dir: Path) -> None:
    repo = MovieRepository(artifacts_dir=str(fake_artifacts_dir))
    repo.load()

    movie = repo.get_movie(100)
    assert movie["title"].startswith("Alpha")

    assert repo.total_movies() == 8
    assert 300 in repo.all_movie_ids()


def test_movie_repository_raises_for_unknown_movie(fake_artifacts_dir: Path) -> None:
    repo = MovieRepository(artifacts_dir=str(fake_artifacts_dir))
    repo.load()

    with pytest.raises(ValueError, match="not found"):
        repo.get_movie(9999)


# ---------------------------------------------------------------------------
# Predictor end-to-end
# ---------------------------------------------------------------------------


def test_predictor_returns_top_k_known_user(fake_artifacts_dir: Path) -> None:
    """Usuário conhecido pelo encoder deve receber top_k reais do modelo."""

    predictor = Predictor(artifacts_dir=str(fake_artifacts_dir))
    predictor.load()

    results = predictor.predict(user_id=10, top_k=3)

    assert len(results) == 3
    for item in results:
        assert 0.0 <= item["predicted_score"] <= 1.0
        assert item["movie_id"] in [100, 200, 300, 400, 500, 600, 700, 800]

    ranks = [r["rank"] for r in results]
    assert ranks == [1, 2, 3]


def test_predictor_returns_empty_list_for_unknown_user(fake_artifacts_dir: Path) -> None:
    """Usuário desconhecido deve retornar lista vazia (aciona fallback no serviço)."""

    predictor = Predictor(artifacts_dir=str(fake_artifacts_dir))
    predictor.load()

    results = predictor.predict(user_id=999, top_k=5)
    assert results == []


def test_predictor_raises_when_not_loaded(fake_artifacts_dir: Path) -> None:
    """Chamar predict() sem load() deve gerar erro claro."""

    predictor = Predictor(artifacts_dir=str(fake_artifacts_dir))

    with pytest.raises(RuntimeError, match="não foi inicializado"):
        predictor.predict(user_id=10, top_k=1)


# ---------------------------------------------------------------------------
# RecommendationService com artefatos carregados
# ---------------------------------------------------------------------------


def test_service_uses_model_when_ready(
    fake_artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Com artefatos carregados no cache, o serviço deve usar o modelo real
    em vez do fallback de popularidade.
    """

    monkeypatch.chdir(fake_artifacts_dir.parent)

    ModelLoader(artifacts_dir=str(fake_artifacts_dir)).load()
    EncoderLoader(artifacts_dir=str(fake_artifacts_dir)).load()
    MovieRepository(artifacts_dir=str(fake_artifacts_dir)).load()

    service = RecommendationService()
    service.predictor = Predictor(artifacts_dir=str(fake_artifacts_dir))

    response = service.recommend(user_id=10, top_k=3)

    assert response.user_id == 10
    assert response.total == 3
    assert len(response.recommendations) == 3
    for item in response.recommendations:
        assert item.movie_id in [100, 200, 300, 400, 500, 600, 700, 800]
