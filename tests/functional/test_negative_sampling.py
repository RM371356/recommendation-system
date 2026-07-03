"""
Testes funcionais do negative sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.negative_sampling import generate_negative_samples

pytestmark = pytest.mark.functional


@pytest.fixture()
def positives_only_df() -> pd.DataFrame:
    """Interações positivas de 3 usuários com 5 filmes."""

    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "movie_idx": [0, 1, 2, 3, 4],
            "label": [1, 1, 1, 1, 1],
        }
    )


def test_generates_at_most_n_negatives_per_user(positives_only_df: pd.DataFrame) -> None:
    """Deve gerar ~n_negatives por usuário e todos com label=0."""

    np.random.seed(42)
    result = generate_negative_samples(positives_only_df, n_negatives=2)

    negatives = result[result["label"] == 0]

    for user_idx in positives_only_df["user_idx"].unique():
        count = (negatives["user_idx"] == user_idx).sum()
        assert count <= 2


def test_negatives_are_not_watched_by_the_user(positives_only_df: pd.DataFrame) -> None:
    """Filme sorteado como negativo não pode estar entre os assistidos pelo user."""

    np.random.seed(0)
    result = generate_negative_samples(positives_only_df, n_negatives=3)
    negatives = result[result["label"] == 0]

    for _, row in negatives.iterrows():
        watched = positives_only_df.loc[
            positives_only_df["user_idx"] == row["user_idx"], "movie_idx"
        ].tolist()
        assert row["movie_idx"] not in watched


def test_all_original_positives_preserved(positives_only_df: pd.DataFrame) -> None:
    """A concatenação não deve descartar nenhum positivo original."""

    np.random.seed(1)
    result = generate_negative_samples(positives_only_df, n_negatives=1)
    positives_in_result = result[result["label"] == 1]

    assert len(positives_in_result) == len(positives_only_df)
