"""
Testes funcionais das métricas de avaliação.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import RecommendationMetrics

pytestmark = pytest.mark.functional


def test_perfect_predictions_yield_score_one() -> None:
    """Se todas as predições estão corretas, todas as métricas devem ser 1.0."""

    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    assert RecommendationMetrics.accuracy(y_true, y_pred) == 1.0
    assert RecommendationMetrics.precision(y_true, y_pred) == 1.0
    assert RecommendationMetrics.recall(y_true, y_pred) == 1.0
    assert RecommendationMetrics.f1(y_true, y_pred) == 1.0


def test_all_wrong_predictions_yield_zero_accuracy() -> None:
    """Predições invertidas devem levar a acurácia para 0."""

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1])

    assert RecommendationMetrics.accuracy(y_true, y_pred) == 0.0
    assert RecommendationMetrics.recall(y_true, y_pred) == 0.0


def test_metrics_are_bounded_between_zero_and_one() -> None:
    """Métricas nunca podem ficar fora de [0, 1]."""

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)

    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        value = getattr(RecommendationMetrics, metric_name)(y_true, y_pred)
        assert 0.0 <= value <= 1.0, f"{metric_name}={value} fora do intervalo"
