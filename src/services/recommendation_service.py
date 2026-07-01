"""
Recommendation Service

Orquestra a geração de recomendações para um usuário.

- Se o usuário for conhecido, utiliza o modelo MLP via Predictor.
- Se o usuário for desconhecido (cold-start), retorna os filmes
  mais populares como fallback.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.api.schemas import (
    RecommendationItem,
    RecommendationResponse,
)
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ARTIFACTS_DIR = "artifacts"


class RecommendationService:

    def __init__(self) -> None:
        self.predictor = Predictor(artifacts_dir=_ARTIFACTS_DIR)
        self._popular_movies: list[dict] | None = None

    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
    ) -> RecommendationResponse:

        recommendations_raw: list[dict] = []

        if self.predictor.is_ready:
            # Tenta gerar recomendações reais com o modelo treinado
            recommendations_raw = self.predictor.predict(
                user_id=user_id,
                top_k=top_k,
            )

        if not recommendations_raw:
            # Fallback: filmes mais populares para cold-start ou modelo não carregado
            logger.warning(
                f"Usando fallback de popularidade para user_id={user_id}."
            )
            recommendations_raw = self._get_popular_movies(top_k=top_k)

        recommendations = [
            RecommendationItem(
                rank=item["rank"],
                movie_id=item["movie_id"],
                title=item["title"],
                genres=item["genres"],
                predicted_score=item["predicted_score"],
            )
            for item in recommendations_raw
        ]

        return RecommendationResponse(
            user_id=user_id,
            total=len(recommendations),
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------

    def _get_popular_movies(
        self,
        top_k: int = 10,
        min_ratings: int = 10,
    ) -> list[dict]:
        """
        Retorna os filmes mais populares com base no número de avaliações
        e na média de rating do dataset de treino/features.

        Carregado apenas uma vez e mantido em memória.
        """

        if self._popular_movies is not None:
            return self._popular_movies[:top_k]

        # Tenta carregar do CSV de features (que já existe após pipeline)
        features_path = Path("data/features/features.csv")
        movies_path = Path(f"{_ARTIFACTS_DIR}/metadata/movies.csv")

        if not features_path.exists() or not movies_path.exists():
            # Fallback absoluto sem dados
            return [
                {
                    "rank": i + 1,
                    "movie_id": i + 1,
                    "title": f"Popular Movie {i + 1}",
                    "genres": "Unknown",
                    "predicted_score": 1.0 - i * 0.01,
                }
                for i in range(top_k)
            ]

        features_df = pd.read_csv(features_path)
        movies_df = pd.read_csv(movies_path)

        # Calcula média e contagem de ratings por movieId
        stats = (
            features_df.groupby("movieId")["label"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stats.columns = ["movieId", "avg_rating", "num_ratings"]

        # Filtra filmes com número mínimo de avaliações
        popular = stats[stats["num_ratings"] >= min_ratings].sort_values(
            "avg_rating", ascending=False
        )

        # Une com metadados dos filmes
        popular = popular.merge(movies_df, on="movieId", how="left")

        self._popular_movies = [
            {
                "rank": rank,
                "movie_id": int(row["movieId"]),
                "title": str(row.get("title", f"Movie {row['movieId']}")),
                "genres": str(row.get("genres", "Unknown")),
                "predicted_score": round(float(row["avg_rating"]), 4),
            }
            for rank, (_, row) in enumerate(popular.head(100).iterrows(), start=1)
        ]

        return self._popular_movies[:top_k]