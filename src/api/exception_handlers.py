from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class RecommendationError(Exception):

    def __init__(
        self,
        message: str,
        status_code: int = 400,
    ):

        self.message = message
        self.status_code = status_code


RecommendationException = RecommendationError


def register_exception_handlers(
    app: FastAPI,
):

    @app.exception_handler(
        RecommendationError
    )
    async def recommendation_exception_handler(
        request: Request,
        exc: RecommendationError,
    ):

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ):

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
            },
        )