from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset() -> None:

    df = pd.read_csv(
        "data/features/features.csv"
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    Path("data/splits").mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        "data/splits/train.csv",
        index=False,
    )

    val_df.to_csv(
        "data/splits/val.csv",
        index=False,
    )

    test_df.to_csv(
        "data/splits/test.csv",
        index=False,
    )


if __name__ == "__main__":
    split_dataset()