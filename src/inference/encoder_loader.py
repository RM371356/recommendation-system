"""
Encoder Loader

Responsável por carregar os LabelEncoders
gerados durante o treinamento.

Os encoders são carregados apenas uma vez
e armazenados no cache.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder

from src.inference.cache import cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EncoderLoader:

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
    ) -> None:

        self.artifacts_dir = Path(artifacts_dir)

        self.encoder_dir = (
            self.artifacts_dir / "encoders"
        )

        self.user_encoder_path = (
            self.encoder_dir / "user_encoder.pkl"
        )

        self.movie_encoder_path = (
            self.encoder_dir / "movie_encoder.pkl"
        )

    # --------------------------------------------------

    def load(self) -> None:
        """
        Carrega os encoders apenas uma vez.
        """

        if (
            cache.user_encoder is not None
            and cache.movie_encoder is not None
        ):

            logger.info(
                "Encoders already loaded."
            )

            return

        logger.info(
            "Loading encoders..."
        )

        cache.user_encoder = self._load_user_encoder()

        cache.movie_encoder = self._load_movie_encoder()

        logger.info(
            "Encoders loaded successfully."
        )

    # --------------------------------------------------

    def _load_user_encoder(
        self,
    ) -> LabelEncoder:

        if not self.user_encoder_path.exists():

            raise FileNotFoundError(
                f"User encoder not found: {self.user_encoder_path}"
            )

        logger.info(
            "Loading user encoder..."
        )

        return joblib.load(
            self.user_encoder_path
        )

    # --------------------------------------------------

    def _load_movie_encoder(
        self,
    ) -> LabelEncoder:

        if not self.movie_encoder_path.exists():

            raise FileNotFoundError(
                f"Movie encoder not found: {self.movie_encoder_path}"
            )

        logger.info(
            "Loading movie encoder..."
        )

        return joblib.load(
            self.movie_encoder_path
        )

    # --------------------------------------------------

    @property
    def user_encoder(self) -> LabelEncoder:

        if cache.user_encoder is None:

            raise RuntimeError(
                "User encoder not loaded."
            )

        return cache.user_encoder

    # --------------------------------------------------

    @property
    def movie_encoder(self) -> LabelEncoder:

        if cache.movie_encoder is None:

            raise RuntimeError(
                "Movie encoder not loaded."
            )

        return cache.movie_encoder

    # --------------------------------------------------

    def encode_user(
        self,
        user_id: int,
    ) -> int:
        """
        Converte user_id -> user_idx
        """

        return int(
            self.user_encoder.transform(
                [user_id]
            )[0]
        )

    # --------------------------------------------------

    def decode_user(
        self,
        user_idx: int,
    ) -> int:
        """
        Converte user_idx -> user_id
        """

        return int(
            self.user_encoder.inverse_transform(
                [user_idx]
            )[0]
        )

    # --------------------------------------------------

    def encode_movie(
        self,
        movie_id: int,
    ) -> int:
        """
        movie_id -> movie_idx
        """

        return int(
            self.movie_encoder.transform(
                [movie_id]
            )[0]
        )

    # --------------------------------------------------

    def decode_movie(
        self,
        movie_idx: int,
    ) -> int:
        """
        movie_idx -> movie_id
        """

        return int(
            self.movie_encoder.inverse_transform(
                [movie_idx]
            )[0]
        )

    # --------------------------------------------------

    def unload(self) -> None:

        logger.info(
            "Removing encoders from cache."
        )

        cache.user_encoder = None

        cache.movie_encoder = None