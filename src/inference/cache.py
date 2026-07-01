"""
Central cache para os artefatos da camada de inferência.

Este módulo mantém em memória todos os objetos pesados
necessários para realizar recomendações.

Objetos armazenados:

- Modelo PyTorch
- User Encoder
- Movie Encoder
- DataFrame de filmes
- Configuração do modelo

Todos são carregados apenas uma vez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


@dataclass
class InferenceCache:
    """
    Cache global utilizado pela API.

    Todos os atributos são inicializados como None e
    preenchidos durante o carregamento dos artefatos.
    """

    model: torch.nn.Module | None = None

    user_encoder: LabelEncoder | None = None

    movie_encoder: LabelEncoder | None = None

    movies: pd.DataFrame | None = None

    config: dict[str, Any] | None = None

    initialized: bool = False

    @property
    def is_ready(self) -> bool:
        """
        Verifica se todos os artefatos foram carregados.
        """

        return (
            self.model is not None
            and self.user_encoder is not None
            and self.movie_encoder is not None
            and self.movies is not None
            and self.config is not None
        )

    def clear(self) -> None:
        """
        Limpa completamente o cache.
        Útil para testes ou recarga dos artefatos.
        """

        self.model = None
        self.user_encoder = None
        self.movie_encoder = None
        self.movies = None
        self.config = None
        self.initialized = False

    def summary(self) -> dict:
        """
        Retorna um resumo do estado atual do cache.
        """

        return {
            "initialized": self.initialized,
            "ready": self.is_ready,
            "model_loaded": self.model is not None,
            "user_encoder_loaded": self.user_encoder is not None,
            "movie_encoder_loaded": self.movie_encoder is not None,
            "movies_loaded": self.movies is not None,
            "config_loaded": self.config is not None,
        }


# --------------------------------------------------------------------
# Singleton utilizado por toda a aplicação
# --------------------------------------------------------------------

cache = InferenceCache()