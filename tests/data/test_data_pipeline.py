"""
Testes de qualidade dos dados.

Validam os splits gerados pelo pipeline (`data/splits/*.csv`):
- Schema consistente
- Sem valores ausentes em colunas críticas
- Sem vazamento entre train/val/test
- Distribuição de labels dentro do razoável

Se os splits ainda não existirem no ambiente, o teste é skipado
para não bloquear o CI numa máquina limpa.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.data

SPLITS_DIR = Path("data/splits")
REQUIRED_COLUMNS = {
    "userId",
    "movieId",
    "rating",
    "label",
    "user_idx",
    "movie_idx",
}


def _load_split(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"Split não encontrado: {path}")
    return pd.read_csv(path)


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_split_has_required_columns(split_name: str) -> None:
    df = _load_split(split_name)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Colunas faltando em {split_name}: {missing}"


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_split_has_no_nulls_in_critical_columns(split_name: str) -> None:
    df = _load_split(split_name)
    for col in ["userId", "movieId", "rating", "label"]:
        assert df[col].notna().all(), f"Nulos encontrados em {split_name}.{col}"


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_split_labels_are_binary(split_name: str) -> None:
    df = _load_split(split_name)
    assert set(df["label"].unique()).issubset({0, 1})


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_split_has_positive_and_negative_labels(split_name: str) -> None:
    """Distribuição minimamente saudável: ambos os labels devem aparecer."""

    df = _load_split(split_name)
    label_counts = df["label"].value_counts()
    assert 0 in label_counts.index, f"{split_name} não tem exemplos negativos"
    assert 1 in label_counts.index, f"{split_name} não tem exemplos positivos"


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_split_rating_within_expected_range(split_name: str) -> None:
    """Ratings MovieLens sempre em [0.5, 5.0]."""

    df = _load_split(split_name)
    assert df["rating"].min() >= 0.5
    assert df["rating"].max() <= 5.0


def test_no_row_leakage_across_splits() -> None:
    """
    Os splits devem ser disjuntos: nenhuma tripla (user, movie, timestamp)
    pode aparecer em dois splits ao mesmo tempo.
    """

    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")

    def _key(df: pd.DataFrame) -> set[tuple]:
        return set(
            map(
                tuple,
                df[["userId", "movieId", "timestamp"]].itertuples(index=False),
            )
        )

    train_keys = _key(train)
    val_keys = _key(val)
    test_keys = _key(test)

    assert train_keys.isdisjoint(val_keys), "Vazamento train ↔ val"
    assert train_keys.isdisjoint(test_keys), "Vazamento train ↔ test"
    assert val_keys.isdisjoint(test_keys), "Vazamento val ↔ test"


def test_train_split_is_the_largest() -> None:
    """
    O split de treino deve ter a maior quantidade de linhas
    (segundo o `split_data.py`: 70/15/15).
    """

    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")

    assert len(train) > len(val)
    assert len(train) > len(test)


# ---------------------------------------------------------------------------
# Data quality on the pipeline itself (in-memory)
# ---------------------------------------------------------------------------


def test_build_features_produces_encoded_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Executa `build_features()` num diretório temporário e verifica o
    schema do CSV resultante.
    """

    from src.features.build_features import build_features

    monkeypatch.chdir(tmp_path)

    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    features_dir = tmp_path / "data" / "features"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 20, 10, 30, 40],
            "rating": [5.0, 4.0, 3.0, 4.5, 2.0],
            "timestamp": [1, 2, 3, 4, 5],
            "label": [1, 1, 0, 1, 0],
        }
    ).to_csv(processed_dir / "ratings_processed.csv", index=False)

    pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["x", "y", "z", "w"],
        }
    ).to_csv(raw_dir / "movies.csv", index=False)

    build_features()

    result = pd.read_csv(features_dir / "features.csv")
    assert {"user_idx", "movie_idx"}.issubset(result.columns)
    assert result["user_idx"].dtype.kind in "iu"
    assert result["movie_idx"].dtype.kind in "iu"

    # Encoders e movies.csv precisam ter sido persistidos
    assert (tmp_path / "artifacts" / "encoders" / "user_encoder.pkl").exists()
    assert (tmp_path / "artifacts" / "encoders" / "movie_encoder.pkl").exists()
    assert (tmp_path / "artifacts" / "metadata" / "movies.csv").exists()
