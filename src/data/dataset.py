
import pandas as pd
import torch
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        self.users = torch.tensor(
            dataframe["user_idx"].values,
            dtype=torch.long,
        )

        self.items = torch.tensor(
            dataframe["movie_idx"].values,
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            dataframe["label"].values,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.users[idx],
            self.items[idx],
            self.labels[idx],
        )