from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(
        ...,
        gt=0,
        description="MovieLens user identifier",
        examples=[1],
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations",
    )


class RecommendationItem(BaseModel):
    rank: int

    movie_id: int

    title: str

    genres: str

    predicted_score: float


class RecommendationResponse(BaseModel):
    user_id: int

    total: int

    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str

    version: str

    model_loaded: bool

    artifacts_loaded: bool