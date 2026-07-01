from functools import lru_cache

from src.services.recommendation_service import RecommendationService


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    """
    Retorna uma única instância do RecommendationService
    durante toda a execução da API.
    """
    return RecommendationService()