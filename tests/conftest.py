"""
Configurações compartilhadas para toda a suíte de testes.

Este conftest fornece fixtures que:
- Isolam o cache global de inferência entre testes (evita contaminação de estado).
- Constroem um TestClient da FastAPI com dependências mockadas.
- Geram artefatos fake (modelo MLP mínimo, encoders, movies.csv) em tmp_path
  quando um teste precisa exercitar o fluxo real de inferência.
- Fornecem datasets sintéticos pequenos para exercitar features e dataset.

Nenhum destes helpers depende de artefatos reais treinados, permitindo que a
suíte inteira seja executada em uma máquina limpa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

from src.inference.cache import cache as global_cache
from src.models.model_factory import ModelFactory


# ---------------------------------------------------------------------------
# Ciclo de vida do cache global
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_inference_cache() -> Iterator[None]:
    """
    Zera o cache singleton antes e depois de cada teste.

    Sem isso, um teste que carrega artefatos deixaria estado residual para o
    próximo — o que quebra o determinismo dos testes de fallback.
    Também invalida o lru_cache do RecommendationService para forçar
    reconstrução limpa a cada teste.
    """

    from src.api.dependencies import get_recommendation_service

    global_cache.clear()
    get_recommendation_service.cache_clear()
    yield
    global_cache.clear()
    get_recommendation_service.cache_clear()


# ---------------------------------------------------------------------------
# TestClient da FastAPI
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client() -> Iterator[TestClient]:
    """
    Cria um TestClient da aplicação principal. O lifespan é executado
    normalmente, mas como o diretório `artifacts/` provavelmente está vazio,
    a API sobe em modo "fallback de popularidade" (comportamento esperado
    e coberto pelos testes de resiliência).
    """

    from src.api.main import app

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Datasets sintéticos
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_ratings_df() -> pd.DataFrame:
    """
    Pequeno dataframe estilo MovieLens usado em testes de preprocess/features.
    Contém interações positivas e negativas de forma balanceada.
    """

    rng = np.random.default_rng(42)
    n = 50

    return pd.DataFrame(
        {
            "userId": rng.integers(1, 10, size=n),
            "movieId": rng.integers(1, 20, size=n),
            "rating": rng.choice([1.0, 2.5, 3.5, 4.0, 4.5, 5.0], size=n),
            "timestamp": rng.integers(1_000_000_000, 1_600_000_000, size=n),
        }
    )


@pytest.fixture()
def encoded_ratings_df(synthetic_ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataframe já com colunas `label`, `user_idx` e `movie_idx`, no formato
    que o `RatingsDataset` espera.
    """

    df = synthetic_ratings_df.copy()
    df["label"] = (df["rating"] >= 4).astype(int)
    df["user_idx"] = LabelEncoder().fit_transform(df["userId"])
    df["movie_idx"] = LabelEncoder().fit_transform(df["movieId"])
    return df


# ---------------------------------------------------------------------------
# Artefatos fake (modelo + encoders + movies.csv)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_artifacts_dir(tmp_path: Path) -> Path:
    """
    Constrói uma pasta `artifacts/` completa em tmp_path com:
      - artifacts/model/best_model.pth   (modelo MLP mínimo)
      - artifacts/model/config.json      (arquitetura para reconstrução)
      - artifacts/encoders/user_encoder.pkl
      - artifacts/encoders/movie_encoder.pkl
      - artifacts/metadata/movies.csv
    Retorna o Path da pasta `artifacts`.
    """

    import json

    n_users, n_items, embedding_dim = 5, 8, 4

    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / "model").mkdir(parents=True)
    (artifacts_dir / "encoders").mkdir(parents=True)
    (artifacts_dir / "metadata").mkdir(parents=True)

    model = ModelFactory.create(
        model_name="mlp",
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
    )
    torch.save(model.state_dict(), artifacts_dir / "model" / "best_model.pth")

    (artifacts_dir / "model" / "config.json").write_text(
        json.dumps(
            {
                "model_name": "mlp",
                "n_users": n_users,
                "n_items": n_items,
                "embedding_dim": embedding_dim,
            }
        )
    )

    user_encoder = LabelEncoder().fit(np.array([10, 20, 30, 40, 50]))
    movie_encoder = LabelEncoder().fit(np.array([100, 200, 300, 400, 500, 600, 700, 800]))
    joblib.dump(user_encoder, artifacts_dir / "encoders" / "user_encoder.pkl")
    joblib.dump(movie_encoder, artifacts_dir / "encoders" / "movie_encoder.pkl")

    movies_df = pd.DataFrame(
        {
            "movieId": [100, 200, 300, 400, 500, 600, 700, 800],
            "title": [
                "Alpha (1999)",
                "Bravo (2000)",
                "Charlie (2001)",
                "Delta (2002)",
                "Echo (2003)",
                "Foxtrot (2004)",
                "Golf (2005)",
                "Hotel (2006)",
            ],
            "genres": [
                "Action",
                "Drama",
                "Comedy",
                "Sci-Fi",
                "Horror",
                "Romance",
                "Thriller",
                "Documentary",
            ],
        }
    )
    movies_df.to_csv(artifacts_dir / "metadata" / "movies.csv", index=False)

    return artifacts_dir
