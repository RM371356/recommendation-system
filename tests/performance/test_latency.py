"""
Testes de performance / latência (soft budgets).

Não substituem um load test real (Locust/k6), mas dão um sinal
precoce de regressões grosseiras. Os limites são propositalmente
generosos para não falhar em máquinas de CI lentas.
"""

from __future__ import annotations

import time

import pytest
import torch
from fastapi.testclient import TestClient

from src.models.model_factory import ModelFactory

pytestmark = pytest.mark.performance


HEALTH_BUDGET_SECONDS = 0.5
RECOMMEND_BUDGET_SECONDS = 2.0
MODEL_FORWARD_BUDGET_SECONDS = 0.5


def _measure(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_health_endpoint_latency(api_client: TestClient) -> None:
    """O /health deve responder em menos de meio segundo."""

    # Warmup
    api_client.get("/api/v1/health")

    elapsed = _measure(lambda: api_client.get("/api/v1/health"))
    assert elapsed < HEALTH_BUDGET_SECONDS, f"{elapsed:.3f}s > {HEALTH_BUDGET_SECONDS}s"


def test_recommend_fallback_latency(api_client: TestClient) -> None:
    """
    Mesmo no modo fallback (sem modelo), a recomendação deve
    responder em tempo aceitável.
    """

    api_client.post("/api/v1/recommend", json={"user_id": 1, "top_k": 5})  # warmup

    elapsed = _measure(
        lambda: api_client.post(
            "/api/v1/recommend", json={"user_id": 1, "top_k": 10}
        )
    )
    assert elapsed < RECOMMEND_BUDGET_SECONDS, f"{elapsed:.3f}s > {RECOMMEND_BUDGET_SECONDS}s"


def test_model_forward_pass_is_fast() -> None:
    """
    Um forward em batch pequeno tem que caber num orçamento
    de tempo bem folgado, mesmo em CPU.
    """

    model = ModelFactory.create(
        model_name="mlp", n_users=100, n_items=200, embedding_dim=32
    )
    model.eval()

    users = torch.randint(0, 100, (256,), dtype=torch.long)
    items = torch.randint(0, 200, (256,), dtype=torch.long)

    with torch.no_grad():
        elapsed = _measure(lambda: model(users, items))

    assert elapsed < MODEL_FORWARD_BUDGET_SECONDS, (
        f"{elapsed:.3f}s > {MODEL_FORWARD_BUDGET_SECONDS}s"
    )
