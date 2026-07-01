"""
FastAPI Application Entry Point

Responsável por:
- Configurar o lifespan (startup/shutdown) da aplicação
- Carregar artefatos (modelo, encoders, metadados) na inicialização
- Registrar exception handlers
- Incluir routers
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routes import router
from src.inference.encoder_loader import EncoderLoader
from src.inference.model_loader import ModelLoader
from src.inference.movie_repository import MovieRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ARTIFACTS_DIR = "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação FastAPI.

    - Startup: carrega modelo, encoders e metadados de filmes em memória.
    - Shutdown: libera os artefatos do cache.
    """

    # ---- Startup --------------------------------------------------------
    logger.info("Starting up Recommendation API...")

    model_loader = ModelLoader(artifacts_dir=_ARTIFACTS_DIR)
    encoder_loader = EncoderLoader(artifacts_dir=_ARTIFACTS_DIR)
    movie_repo = MovieRepository(artifacts_dir=_ARTIFACTS_DIR)

    try:
        model_loader.load()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(
            f"Model artifacts not found: {exc}. "
            "API will start without model — using popularity fallback."
        )

    try:
        encoder_loader.load()
        logger.info("Encoders loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(
            f"Encoder artifacts not found: {exc}. "
            "API will start without encoders — using popularity fallback."
        )

    try:
        movie_repo.load()
        logger.info("Movie metadata loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(
            f"Movie metadata not found: {exc}. "
            "API will start without movie metadata."
        )

    logger.info("Startup complete.")

    yield

    # ---- Shutdown -------------------------------------------------------
    logger.info("Shutting down Recommendation API...")
    model_loader.unload()
    encoder_loader.unload()
    movie_repo.unload()
    logger.info("Cache cleared.")


app = FastAPI(
    title="Recommendation API",
    description="API de recomendação de filmes baseada em MLP com embeddings.",
    version="2.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)

app.include_router(router)