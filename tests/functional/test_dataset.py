"""
Testes funcionais do RatingsDataset (torch.utils.data.Dataset).
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.data.dataset import RatingsDataset

pytestmark = pytest.mark.functional


def test_dataset_length_matches_dataframe(encoded_ratings_df: pd.DataFrame) -> None:
    dataset = RatingsDataset(encoded_ratings_df)
    assert len(dataset) == len(encoded_ratings_df)


def test_dataset_returns_tuple_of_three_tensors(encoded_ratings_df: pd.DataFrame) -> None:
    """__getitem__ deve retornar (user, item, label) — tensores escalares."""

    dataset = RatingsDataset(encoded_ratings_df)
    user, item, label = dataset[0]

    assert isinstance(user, torch.Tensor)
    assert isinstance(item, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert user.dtype == torch.long
    assert item.dtype == torch.long
    assert label.dtype == torch.float32


def test_dataset_labels_match_source_dataframe(encoded_ratings_df: pd.DataFrame) -> None:
    """Os valores retornados devem bater com o dataframe de origem."""

    dataset = RatingsDataset(encoded_ratings_df)

    for idx in range(min(5, len(dataset))):
        user, item, label = dataset[idx]
        assert user.item() == encoded_ratings_df.iloc[idx]["user_idx"]
        assert item.item() == encoded_ratings_df.iloc[idx]["movie_idx"]
        assert label.item() == pytest.approx(
            float(encoded_ratings_df.iloc[idx]["label"])
        )
