"""
Predictor

Responsável por realizar inferência em lote para um usuário,
rankeando todos os filmes candidatos e retornando os top-K.
"""

from __future__ import annotations

import numpy as np
import torch

from src.inference.cache import cache
from src.inference.encoder_loader import EncoderLoader
from src.inference.model_loader import ModelLoader
from src.inference.movie_repository import MovieRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Realiza recomendações utilizando o modelo treinado carregado em memória.

    Fluxo:
    1. Codifica o user_id em user_idx via LabelEncoder.
    2. Obtém todos os movie_ids disponíveis e filtra os já assistidos.
    3. Executa inferência em lote para todos os candidatos.
    4. Ordena por score e retorna os top-K com metadados.
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
    ) -> None:
        self.model_loader = ModelLoader(artifacts_dir=artifacts_dir)
        self.encoder_loader = EncoderLoader(artifacts_dir=artifacts_dir)
        self.movie_repo = MovieRepository(artifacts_dir=artifacts_dir)

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Carrega todos os artefatos necessários para inferência."""

        self.model_loader.load()
        self.encoder_loader.load()
        self.movie_repo.load()

        logger.info("Predictor ready.")

    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return cache.is_ready

    # ------------------------------------------------------------------

    def predict(
        self,
        user_id: int,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Gera as top_k recomendações para um usuário.

        Args:
            user_id: Identificador original do usuário (MovieLens ID).
            top_k: Número de recomendações a retornar.

        Returns:
            Lista de dicts com chaves: movie_id, title, genres, score.
        """

        if not self.is_ready:
            raise RuntimeError(
                "Predictor não foi inicializado. Chame load() antes de predict()."
            )

        # --- Codifica o user_id -----------------------------------------
        try:
            user_idx = self.encoder_loader.encode_user(user_id)
        except ValueError:
            # Usuário desconhecido: não faz predição, retorna lista vazia
            logger.warning(
                f"user_id={user_id} não encontrado no encoder. "
                "Fallback de popularidade será aplicado pelo serviço."
            )
            return []

        # --- Obtém todos os filmes e filtra os já assistidos --------------
        all_movie_ids = self.movie_repo.all_movie_ids()

        # Filtra apenas movie_ids que possuem índice no encoder.
        # Usa um set para busca O(1) — evita custo quadrático quando há
        # milhares de filmes candidatos.
        known_movie_ids = set(cache.movie_encoder.classes_.tolist())
        candidate_movie_ids = [m for m in all_movie_ids if m in known_movie_ids]

        # --- Batch inference ----------------------------------------------
        # Codifica todos os candidatos numa única chamada de transform,
        # em vez de uma por filme (muito mais rápido para catálogos grandes).
        movie_indices = cache.movie_encoder.transform(candidate_movie_ids).astype(
            np.int64
        )

        user_tensor = torch.tensor(
            [user_idx] * len(movie_indices),
            dtype=torch.long,
        )
        item_tensor = torch.tensor(
            movie_indices,
            dtype=torch.long,
        )

        model = cache.model
        model.eval()

        with torch.no_grad():
            logits = model(user_tensor, item_tensor).squeeze()
            scores = torch.sigmoid(logits).numpy()

        if scores.ndim == 0:
            scores = scores.reshape(-1)

        # --- Ranking top-K -----------------------------------------------
        top_k_actual = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k_actual]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            movie_id = candidate_movie_ids[idx]
            score = float(scores[idx])

            try:
                movie_meta = self.movie_repo.get_movie(movie_id)
            except ValueError:
                movie_meta = {
                    "movie_id": movie_id,
                    "title": f"Movie {movie_id}",
                    "genres": "Unknown",
                }

            results.append(
                {
                    "rank": rank,
                    "movie_id": movie_meta["movie_id"],
                    "title": movie_meta["title"],
                    "genres": movie_meta["genres"],
                    "predicted_score": score,
                }
            )

        return results
