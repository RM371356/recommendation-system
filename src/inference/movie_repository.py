"""
Movie Repository

Responsável pelo acesso aos metadados dos filmes.

Nenhuma outra classe deve ler movies.csv diretamente.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference.cache import cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MovieRepository:

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
    ) -> None:

        self.artifacts_dir = Path(artifacts_dir)

        self.metadata_dir = (
            self.artifacts_dir / "metadata"
        )

        self.movies_path = (
            self.metadata_dir / "movies.csv"
        )

    # --------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Carrega movies.csv apenas uma vez.
        """

        if cache.movies is not None:

            logger.info(
                "Movies already loaded."
            )

            return cache.movies

        if not self.movies_path.exists():

            raise FileNotFoundError(
                f"Movies metadata not found: {self.movies_path}"
            )

        logger.info(
            "Loading movies metadata..."
        )

        dataframe = pd.read_csv(
            self.movies_path
        )

        required_columns = {
            "movieId",
            "title",
            "genres",
        }

        missing = required_columns.difference(
            dataframe.columns
        )

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        cache.movies = dataframe

        logger.info(
            f"{len(dataframe):,} movies loaded."
        )

        return dataframe

    # --------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:

        if cache.movies is None:

            self.load()

        return cache.movies

    # --------------------------------------------------

    def get_movie(
        self,
        movie_id: int,
    ) -> dict:

        movie = self.dataframe.loc[
            self.dataframe["movieId"] == movie_id
        ]

        if movie.empty:

            raise ValueError(
                f"Movie {movie_id} not found."
            )

        row = movie.iloc[0]

        return {
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": str(row["genres"]),
        }

    # --------------------------------------------------

    def get_movies(
        self,
        movie_ids: list[int],
    ) -> list[dict]:

        movies = self.dataframe[
            self.dataframe["movieId"].isin(
                movie_ids
            )
        ]

        result = []

        for _, row in movies.iterrows():

            result.append(
                {
                    "movie_id": int(row["movieId"]),
                    "title": str(row["title"]),
                    "genres": str(row["genres"]),
                }
            )

        return result

    # --------------------------------------------------

    def exists(
        self,
        movie_id: int,
    ) -> bool:

        return (
            self.dataframe["movieId"]
            == movie_id
        ).any()

    # --------------------------------------------------

    def total_movies(
        self,
    ) -> int:

        return len(
            self.dataframe
        )

    # --------------------------------------------------

    def search(
        self,
        text: str,
    ) -> list[dict]:

        movies = self.dataframe[
            self.dataframe["title"]
            .str.contains(
                text,
                case=False,
                na=False,
            )
        ]

        result = []

        for _, row in movies.iterrows():

            result.append(
                {
                    "movie_id": int(row["movieId"]),
                    "title": str(row["title"]),
                    "genres": str(row["genres"]),
                }
            )

        return result

    # --------------------------------------------------

    def all_movie_ids(
        self,
    ) -> list[int]:

        return (
            self.dataframe["movieId"]
            .astype(int)
            .tolist()
        )

    # --------------------------------------------------

    def unload(self):

        logger.info(
            "Removing movies from cache."
        )

        cache.movies = None