"""
Testes funcionais do preprocess (binarização de ratings).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import RatingBinarizer

pytestmark = pytest.mark.functional


@pytest.mark.parametrize(
    "rating, expected_label",
    [
        (0.5, 0),
        (1.0, 0),
        (2.5, 0),
        (3.9, 0),
        (4.0, 1),
        (4.5, 1),
        (5.0, 1),
    ],
)
def test_binarizer_threshold_at_four(rating: float, expected_label: int) -> None:
    """A regra de negócio é: rating >= 4 vira 1, senão 0."""

    df = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [rating]})
    result = RatingBinarizer().process(df)

    assert result.loc[0, "label"] == expected_label


def test_binarizer_does_not_mutate_input(synthetic_ratings_df: pd.DataFrame) -> None:
    """O RatingBinarizer deve trabalhar em cópia — não pode alterar o input."""

    original = synthetic_ratings_df.copy()
    _ = RatingBinarizer().process(synthetic_ratings_df)

    pd.testing.assert_frame_equal(synthetic_ratings_df, original)


def test_binarizer_output_shape(synthetic_ratings_df: pd.DataFrame) -> None:
    """Deve manter o número de linhas e adicionar a coluna `label`."""

    result = RatingBinarizer().process(synthetic_ratings_df)

    assert len(result) == len(synthetic_ratings_df)
    assert "label" in result.columns
    assert result["label"].isin([0, 1]).all()
