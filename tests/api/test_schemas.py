"""
Testes de contrato dos schemas Pydantic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    HealthResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)

pytestmark = pytest.mark.api


# ---------------------------------------------------------------------------
# RecommendationRequest
# ---------------------------------------------------------------------------


def test_request_accepts_valid_payload() -> None:
    req = RecommendationRequest(user_id=1, top_k=10)
    assert req.user_id == 1
    assert req.top_k == 10


def test_request_default_top_k_is_ten() -> None:
    req = RecommendationRequest(user_id=42)
    assert req.top_k == 10


@pytest.mark.parametrize("invalid_user_id", [0, -1, -999])
def test_request_rejects_non_positive_user_id(invalid_user_id: int) -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(user_id=invalid_user_id)


@pytest.mark.parametrize("invalid_top_k", [0, -5, 101, 1000])
def test_request_rejects_top_k_out_of_range(invalid_top_k: int) -> None:
    """top_k tem que estar em [1, 100]."""

    with pytest.raises(ValidationError):
        RecommendationRequest(user_id=1, top_k=invalid_top_k)


def test_request_rejects_non_integer_user_id() -> None:
    """Strings/floats não conversíveis para int devem ser rejeitadas."""

    with pytest.raises(ValidationError):
        RecommendationRequest(user_id="abc", top_k=5)


# ---------------------------------------------------------------------------
# RecommendationResponse / RecommendationItem
# ---------------------------------------------------------------------------


def test_recommendation_item_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        RecommendationItem(rank=1, movie_id=10)


def test_recommendation_response_accepts_empty_list() -> None:
    resp = RecommendationResponse(user_id=1, total=0, recommendations=[])
    assert resp.total == 0
    assert resp.recommendations == []


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


def test_health_response_accepts_valid_flags() -> None:
    """HealthResponse deve aceitar flags booleanas legítimas."""

    resp = HealthResponse(
        status="ok", version="1", model_loaded=True, artifacts_loaded=False
    )
    assert resp.status == "ok"
    assert resp.model_loaded is True
    assert resp.artifacts_loaded is False


def test_health_response_rejects_non_boolean_flags() -> None:
    """
    Valores que o Pydantic não consegue coagir para bool devem falhar
    (ex.: string arbitrária).
    """

    with pytest.raises(ValidationError):
        HealthResponse(
            status="ok",
            version="1",
            model_loaded="not-a-bool",
            artifacts_loaded=True,
        )
