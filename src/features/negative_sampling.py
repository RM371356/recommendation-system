import numpy as np
import pandas as pd


def generate_negative_samples(
    df: pd.DataFrame,
    n_negatives: int = 2,
):

    users = df["user_idx"].unique()

    movies = set(
        df["movie_idx"].unique()
    )

    negatives = []

    for user in users:

        watched = set(
            df.loc[
                df["user_idx"] == user,
                "movie_idx",
            ]
        )

        candidates = list(
            movies - watched
        )

        sampled = np.random.choice(
            candidates,
            size=min(
                n_negatives,
                len(candidates),
            ),
            replace=False,
        )

        for movie in sampled:

            negatives.append(
                [
                    user,
                    movie,
                    0,
                ]
            )

    negative_df = pd.DataFrame(
        negatives,
        columns=[
            "user_idx",
            "movie_idx",
            "label",
        ],
    )

    return pd.concat(
        [df, negative_df],
        ignore_index=True,
    )