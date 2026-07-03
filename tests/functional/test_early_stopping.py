"""
Testes funcionais do EarlyStopping.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.model_factory import ModelFactory
from src.training.early_stopping import EarlyStopping

pytestmark = pytest.mark.functional


@pytest.fixture()
def dummy_model():
    return ModelFactory.create(model_name="mlp", n_users=2, n_items=2, embedding_dim=2)


def test_improving_loss_never_stops(tmp_path: Path, monkeypatch, dummy_model) -> None:
    """Enquanto o loss melhora, o EarlyStopping não deve pedir parada."""

    monkeypatch.chdir(tmp_path)

    stopping = EarlyStopping(patience=3)
    for loss in [1.0, 0.8, 0.5, 0.3, 0.2]:
        should_stop = stopping.step(loss, dummy_model)
        assert should_stop is False

    assert Path("models/best_model.pth").exists()


def test_stops_after_patience_exhausted(tmp_path: Path, monkeypatch, dummy_model) -> None:
    """Sem melhora por N passos consecutivos, deve pedir parada."""

    monkeypatch.chdir(tmp_path)

    stopping = EarlyStopping(patience=2)
    assert stopping.step(1.0, dummy_model) is False  # baseline
    assert stopping.step(1.5, dummy_model) is False  # counter=1
    assert stopping.step(1.7, dummy_model) is True   # counter=2 -> stop


def test_saves_best_model_only_on_improvement(
    tmp_path: Path, monkeypatch, dummy_model
) -> None:
    """O checkpoint só é sobrescrito quando o loss melhora."""

    monkeypatch.chdir(tmp_path)
    stopping = EarlyStopping(patience=5)

    stopping.step(1.0, dummy_model)
    first_mtime = Path("models/best_model.pth").stat().st_mtime_ns

    stopping.step(1.5, dummy_model)
    second_mtime = Path("models/best_model.pth").stat().st_mtime_ns

    assert first_mtime == second_mtime, "Arquivo não deveria ter sido reescrito"
