"""
Model Loader

Responsável por:

- Localizar o modelo treinado
- Reconstruir a arquitetura
- Carregar os pesos
- Colocar o modelo em modo avaliação
- Armazenar no cache
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.inference.cache import cache
from src.models.model_factory import ModelFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
    ) -> None:

        self.artifacts_dir = Path(artifacts_dir)

        self.model_dir = self.artifacts_dir / "model"

        self.model_path = self.model_dir / "best_model.pth"

        self.config_path = self.model_dir / "config.json"

    # -----------------------------------------------------

    def load(self) -> torch.nn.Module:
        """
        Carrega o modelo em memória.

        Caso já esteja carregado,
        retorna o mesmo objeto.
        """

        if cache.model is not None:

            logger.info(
                "Model already loaded."
            )

            return cache.model

        logger.info(
            "Loading recommendation model..."
        )

        config = self._load_config()

        model = self._build_model(
            config,
        )

        state_dict = torch.load(
            self.model_path,
            map_location="cpu",
        )

        model.load_state_dict(
            state_dict,
        )

        model.eval()

        cache.model = model

        cache.config = config

        logger.info(
            "Model loaded successfully."
        )

        return model

    # -----------------------------------------------------

    def _load_config(self) -> dict:

        if not self.config_path.exists():

            raise FileNotFoundError(
                f"Config not found: {self.config_path}"
            )

        with open(
            self.config_path,
            encoding="utf8",
        ) as file:

            return json.load(file)

    # -----------------------------------------------------

    def _build_model(
        self,
        config: dict,
    ):

        logger.info(
            "Building model architecture..."
        )

        return ModelFactory.create(
            model_name=config["model_name"],
            n_users=config["n_users"],
            n_items=config["n_items"],
            embedding_dim=config["embedding_dim"],
        )

    # -----------------------------------------------------

    @property
    def is_loaded(self) -> bool:

        return cache.model is not None

    # -----------------------------------------------------

    def unload(self):

        logger.info(
            "Unloading model..."
        )

        cache.model = None