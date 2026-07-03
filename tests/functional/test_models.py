"""
Testes funcionais dos modelos (MLPRecommender e ModelFactory).
"""

from __future__ import annotations

import pytest
import torch

from src.models.mlp_model import MLPRecommender
from src.models.model_factory import ModelFactory

pytestmark = pytest.mark.functional


def test_mlp_forward_output_shape() -> None:
    """Forward de MLP com batch de N pares deve retornar tensor (N, 1)."""

    model = MLPRecommender(n_users=10, n_items=15, embedding_dim=8)
    users = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    items = torch.tensor([10, 11, 12, 13], dtype=torch.long)

    output = model(users, items)

    assert output.shape == (4, 1)
    assert output.dtype == torch.float32


def test_mlp_forward_is_deterministic_in_eval_mode() -> None:
    """Em modo eval, o dropout deve ser desligado -> outputs idênticos."""

    model = MLPRecommender(n_users=5, n_items=5, embedding_dim=4)
    model.eval()
    users = torch.tensor([0], dtype=torch.long)
    items = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        first = model(users, items)
        second = model(users, items)

    assert torch.allclose(first, second)


def test_model_factory_creates_mlp() -> None:
    """A factory deve instanciar um MLPRecommender quando model_name='mlp'."""

    model = ModelFactory.create(
        model_name="mlp",
        n_users=3,
        n_items=4,
        embedding_dim=2,
    )
    assert isinstance(model, MLPRecommender)


def test_model_factory_rejects_unknown_model() -> None:
    """Modelo desconhecido deve gerar ValueError informativo."""

    with pytest.raises(ValueError, match="not supported"):
        ModelFactory.create(model_name="xgboost", n_users=1, n_items=1, embedding_dim=1)


def test_mlp_embeddings_are_trainable() -> None:
    """As embeddings devem estar registradas como parâmetros treináveis."""

    model = MLPRecommender(n_users=4, n_items=6, embedding_dim=3)
    trainable_params = {name for name, p in model.named_parameters() if p.requires_grad}

    assert "user_embedding.weight" in trainable_params
    assert "item_embedding.weight" in trainable_params
