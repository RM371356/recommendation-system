import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_features() -> None:
    """
        build_features é a função responsável por construir as features necessárias para o treinamento do modelo de recomendação. 
        Ela lê os dados processados de avaliações a partir do arquivo "ratings_processed.csv", codifica os índices dos usuários e filmes usando LabelEncoder, e salva o DataFrame resultante em um novo arquivo CSV chamado "features.csv".
        Args:
            None: Esta função não recebe argumentos, pois os caminhos dos arquivos de entrada e saída são fixos.
        Returns:
            None: O resultado da construção das features é salvo em um arquivo CSV, e a função não retorna nenhum valor.
    """
    # Lê os dados processados de avaliações a partir do arquivo "ratings_processed.csv" usando o pandas, criando um DataFrame que contém as informações de avaliações dos usuários para os filmes, incluindo as colunas "userId", "movieId", "rating" e "label".
    df = pd.read_csv("data/processed/ratings_processed.csv")

    # Cria instâncias de LabelEncoder para codificar os índices dos usuários e filmes, permitindo que os identificadores originais sejam convertidos em índices numéricos que podem ser usados como entradas para o modelo de recomendação.
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    # Codifica os índices dos usuários e filmes usando LabelEncoder, criando novas colunas "user_idx" e "movie_idx" no DataFrame que contêm os índices codificados correspondentes aos usuários e filmes.
    df["user_idx"] = user_encoder.fit_transform(df["userId"])
    df["movie_idx"] = movie_encoder.fit_transform(df["movieId"])

    # Salva o DataFrame resultante em um novo arquivo CSV chamado "features.csv", sem incluir o índice do DataFrame no arquivo de saída.
    df.to_csv("data/features/features.csv", index=False)

    # Cria diretórios para os encoders e metadados na pasta de artefatos
    encoders_dir = Path("artifacts/encoders")
    encoders_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path("artifacts/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Persiste os encoders gerados no treinamento
    joblib.dump(user_encoder, encoders_dir / "user_encoder.pkl")
    joblib.dump(movie_encoder, encoders_dir / "movie_encoder.pkl")

    # Copia o arquivo movies.csv para a pasta de metadados para uso na API
    raw_movies_path = Path("data/raw/movies.csv")
    if raw_movies_path.exists():
        shutil.copy(raw_movies_path, metadata_dir / "movies.csv")


if __name__ == "__main__":
    build_features()