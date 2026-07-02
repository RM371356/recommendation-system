from fastapi import APIRouter, Depends

from src.api.dependencies import get_recommendation_service
from src.api.schemas import (
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from src.inference.cache import cache
from src.services.recommendation_service import RecommendationService

router = APIRouter(
    prefix="/api/v1",
    tags=["Recommendation"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=cache.model is not None,
        artifacts_loaded=cache.is_ready,
    )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
    service: RecommendationService = Depends(
        get_recommendation_service,
    ),
):

    return service.recommend(
        user_id=request.user_id,
        top_k=request.top_k,
    )