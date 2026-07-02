"""
Testes funcionais do utilitário de seed e do InferenceCache.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.inference.cache import InferenceCache
from src.utils.seed import set_seed

pytestmark = pytest.mark.functional


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


def test_set_seed_produces_reproducible_python_random() -> None:
    """Duas execuções após set_seed(N) devem gerar a mesma sequência."""

    set_seed(123)
    seq_a = [random.random() for _ in range(5)]
    set_seed(123)
    seq_b = [random.random() for _ in range(5)]

    assert seq_a == seq_b


def test_set_seed_produces_reproducible_numpy() -> None:
    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    b = np.random.rand(3)

    np.testing.assert_array_equal(a, b)


def test_set_seed_produces_reproducible_torch() -> None:
    set_seed(99)
    a = torch.rand(3)
    set_seed(99)
    b = torch.rand(3)

    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# InferenceCache
# ---------------------------------------------------------------------------


def test_fresh_cache_is_not_ready() -> None:
    """Cache recém-instanciado tem todos os campos vazios."""

    cache = InferenceCache()

    assert cache.is_ready is False
    assert cache.model is None
    assert cache.user_encoder is None
    assert cache.movie_encoder is None


def test_cache_summary_reflects_partial_state() -> None:
    """summary() deve refletir estado atual do cache."""

    cache = InferenceCache()
    cache.model = object()  # noqa: mock parcial

    summary = cache.summary()
    assert summary["model_loaded"] is True
    assert summary["user_encoder_loaded"] is False
    assert summary["ready"] is False


def test_cache_clear_zera_tudo() -> None:
    """clear() deve restaurar o cache ao estado inicial."""

    cache = InferenceCache()
    cache.model = object()
    cache.movies = object()
    cache.config = {"a": 1}
    cache.initialized = True

    cache.clear()

    assert cache.model is None
    assert cache.movies is None
    assert cache.config is None
    assert cache.initialized is False
